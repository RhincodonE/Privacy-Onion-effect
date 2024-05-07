
from argparse import ArgumentParser
from typing import List
import time
import re
import logging
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision
from torchvision import datasets, transforms
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf
import pandas as pd
from sklearn.metrics import roc_auc_score
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
import os
from torch.utils.data import Subset
from scipy.stats import norm
from torch.utils.data import DataLoader
Section('training', 'Hyperparameters').params(
    batch_size=Param(int, 'Batch size', default=512),
)

Section('data', 'data related stuff').params(
    model_folder=Param(str, 'folder that stores trained experiment model', required=True),
    attacking_results_folder=Param(str, 'folder to store the attacking result', required=True),
    dataset_folder=Param(str, 'folder that store the in and out dataset', required=True),
    statistic_in=Param(str, 'Gaussian distribution parameters for in samples', required=True),
    statistic_out=Param(str, 'Gaussian distribution parameters for out samples', required=True),
    gpu=Param(int, 'GPU to use', required=True),
    num_workers=Param(int, 'The number of workers', default=8),
)



@param('data.gpu')
def make_dataloaders(in_dataset=None, out_dataset=None, val_dataset=None, batch_size=1, num_workers=3, gpu=0):
    paths = {
        'train_in': in_dataset,
        'train_out': out_dataset,
        'test': val_dataset
    }

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    full_loader = torchvision.datasets.CIFAR10('./data/tmp_shadow', train=True, download=True)

    for name in ['train_in','train_out','test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), ToDevice(ch.device(f'cuda:{gpu}')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
        image_pipeline.extend([
            ToTensor(),
            ToDevice(ch.device(f'cuda:{gpu}'), non_blocking=True),
            ToTorchImage(),
            Convert(ch.float16),
            torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])


        loaders[name] = Loader(paths[name], batch_size=batch_size, num_workers=num_workers,
                               order=OrderOption.SEQUENTIAL,pipelines={'image': image_pipeline, 'label': label_pipeline})

    return loaders



def extract_integer_from_path(path):
    # This regular expression pattern finds one or more digits (\d+)
    match = re.search(r'\d+', path)
    if match:
        return int(match.group())  # Convert the matched string to an integer
    return None  # Return None if no integer is found

@param('data.statistic_in')
@param('data.statistic_out')
def load_statistics(statistic_in,statistic_out):
    """Load statistics from a CSV file."""
    stats_df_in = pd.read_csv(statistic_in)
    stats_df_out = pd.read_csv(statistic_out)
    in_mu = stats_df_in.loc[stats_df_in['Parameter'] == 'mu', 'Value'].values[0]
    in_std = stats_df_in.loc[stats_df_in['Parameter'] == 'std', 'Value'].values[0]
    out_mu = stats_df_out.loc[stats_df_out['Parameter'] == 'mu', 'Value'].values[0]
    out_std = stats_df_out.loc[stats_df_out['Parameter'] == 'std', 'Value'].values[0]
    return (in_mu, in_std), (out_mu, out_std)

def check_lists(train_in, train_out):
    # Convert lists to sets
    set_train_in = set(train_in)
    set_train_out = set(train_out)

    # Check for overlap
    overlap = set_train_in.intersection(set_train_out)
    has_overlap = bool(overlap)  # True if there is overlap, False otherwise

    # Check for duplicates within each list
    duplicates_in = len(train_in) != len(set_train_in)
    duplicates_out = len(train_out) != len(set_train_out)

    return has_overlap, duplicates_in, duplicates_out, overlap

@param('data.gpu')
def cal_p(model, in_dataset_index, out_dataset_index, loaders, gpu):
    logits = []
    index = 0
    in_index = np.load(in_dataset_index)
    out_index = np.load(out_dataset_index)

    indices = {'train_in':in_index.tolist(),'train_out':out_index.tolist()}

    # Checking lists
    has_overlap, duplicates_in, duplicates_out, overlap_elements = check_lists(indices['train_in'], indices['train_out'])
    print(f"Is there overlap? {'Yes' if has_overlap else 'No'}")
    print(f"Overlap elements: {overlap_elements if has_overlap else 'None'}")
    print(f"Does 'train_in' have duplicates? {'Yes' if duplicates_in else 'No'}")
    print(f"Does 'train_out' have duplicates? {'Yes' if duplicates_out else 'No'}")

    stat_in,stat_out = load_statistics()
    print(stat_in)
    print(stat_out)


    if model is not None:
        model.eval()
        for name in ['train_in','train_out']:
            with ch.no_grad():
                for ims, labs in tqdm(loaders[name]):
                    out = model(ims.to(f'cuda:{gpu}'))
                    logit = out.softmax(dim=1)
                    values, indice = ch.topk(logit, 2, dim=1)
                    logit_gap = values[:, 0] - values[:, 1]
                    prediction = classify_sample(logit_gap.cpu().item(),stat_in,stat_out)
                    logits.append({"index": indices[name][index], "True_label": name, "Prediction": prediction})
                    index = index+1
            index = 0
        calculate_accuracy(logits)
        calculate_auc(logits)
    return logits

def calculate_accuracy(logits):
    """Calculate the accuracy of predictions."""
    # Count the number of correct predictions
    correct_count = sum(1 for logit in logits if logit['True_label'] == logit['Prediction'])
    # Total number of predictions
    total_count = len(logits)
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f'Acc:{accuracy}')


def calculate_auc(logits):
    # Ensure all predictions are scores or probabilities between 0 and 1
    true_labels = [1 if logit['True_label'] == 'train_in' else 0 for logit in logits]

    # Assuming your classify_sample outputs a probability or score for being 'train_in'
    predicted_scores = [1 if logit['Prediction'] == 'train_in' else 0 for logit in logits]

    if len(set(predicted_scores)) > 1:  # More than one unique score
        auc_score = roc_auc_score(true_labels, predicted_scores)
    else:
        auc_score = 0.5  # If all scores are the same, the AUC is undefined; assume 0.5
    print(f'Auc:{auc_score}')

def pdf_normal(x, mu, sigma):
    """Calculate the normal distribution's PDF at x."""
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def classify_sample(x, in_stats, out_stats):
    """Classify a sample x based on in and out distribution statistics."""
    in_pdf = pdf_normal(x, *in_stats)  # Unpack the tuple in_stats into mu and sigma
    out_pdf = pdf_normal(x, *out_stats)

    if in_pdf > out_pdf:
        return "train_in"
    else:
        return "train_out"


# Model (from KakaoBrain: https://github.com/wbaek/torchskeleton)
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

@param('data.gpu')
def construct_model(gpu = 0):
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda(gpu).half()
    return model

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")
    else:
        logging.info(f"Directory already exists: {path}")

def load_model(model_path, device):
    model = construct_model()  # Assuming this function correctly constructs the model
    model.load_state_dict(ch.load(model_path, map_location=device))
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Loads from args.config_file if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    ensure_directory(config['data.attacking_results_folder'])

    for model_filename in tqdm(os.listdir(config['data.model_folder']), desc=f'Attacking'):
        model_number = extract_integer_from_path(model_filename)

        if model_filename.endswith(('.pt', '.pth')):
            model_path = os.path.join(config['data.model_folder'], model_filename)
            model = load_model(model_path, f'cuda:{config["data.gpu"]}')

            out_index_file_path = os.path.join(config['data.dataset_folder'], f'out_{model_number}.npy')
            in_index_file_path = os.path.join(config['data.dataset_folder'], f'in_{model_number}.npy')

            in_set = os.path.join(config['data.dataset_folder'], f'cifar_in_{model_number}.beton')
            out_set = os.path.join(config['data.dataset_folder'], f'cifar_out_{model_number}.beton')
            test_set = os.path.join(config['data.dataset_folder'], 'cifar_test.beton')


            logging.info(f"Processing model number: {model_number}")
            results_file_path = os.path.join(config['data.attacking_results_folder'], f'result_{model_number}.csv')
            loaders = make_dataloaders(in_dataset=in_set, out_dataset=out_set, val_dataset=test_set)

            logits = cal_p(model=model,in_dataset_index=in_index_file_path,out_dataset_index=out_index_file_path,loaders=loaders)
            pd.DataFrame(logits).to_csv(results_file_path, index=False)
