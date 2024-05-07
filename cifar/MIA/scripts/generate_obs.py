
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
    num_workers=Param(int, 'The number of workers', default=8),
    in_model_save_path=Param(str, 'model save addr', default=True),
)

Section('data', 'data related stuff').params(
    results_folder=Param(str, 'folder to store the observations', required=True),
    in_dataset=Param(str, 'addr of in dataset', required=True),
    out_dataset=Param(str, 'addr of out dataset', required=True),
    val_dataset=Param(str, 'addr of val dataset', required=True),
    out_dataset_index=Param(str, 'addr of index of out dataset', required=True),
    in_dataset_index=Param(str, 'addr of index of in dataset', required=True),
    gpu=Param(int, 'GPU to use', required=True),
)




@param('data.in_dataset')
@param('data.out_dataset')
@param('data.val_dataset')
@param('training.num_workers')
@param('data.gpu')
def make_dataloaders(in_dataset=None, out_dataset=None, val_dataset=None, batch_size=1, num_workers=None, gpu=0):
    paths = {
        'train_in': in_dataset,
        'train_out': out_dataset,
        'test': val_dataset
    }
    print(paths)

    start_time = time.time()
    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    loaders = {}

    for name in ['train_in','train_out', 'test']:
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
                               order=OrderOption.SEQUENTIAL,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})




    return loaders,start_time

def custom_norm_fit(data):
    """
    Fit a normal distribution to the data and return the mean and standard deviation.

    :param data: List or array-like object containing the data to fit.
    :return: A tuple containing the mean and standard deviation of the fitted normal distribution.
    """
    # Check if data is a list of tensors and ensure it's not empty


    # Calculate the mean and standard deviation
    mu = np.mean(data)
    std = np.std(data, ddof=1)  # Using ddof=1 for sample standard deviation
    return mu, std


def extract_integer_from_path(path):
    # This regular expression pattern finds one or more digits (\d+)
    match = re.search(r'\d+', path)
    if match:
        return int(match.group())  # Convert the matched string to an integer
    return None  # Return None if no integer is found


@param('data.gpu')
def cal_p(model, in_dataset_index, out_dataset_index, loaders, gpu):
    logits = []
    index = 0

    indices = {'train_in':np.load(in_dataset_index),'train_out':np.load(out_dataset_index)}

    if model is not None:
        model.eval()
        for name in ['train_in','train_out','test']:
            with ch.no_grad():
                for ims, labs in tqdm(loaders[name]):
                    out = model(ims)
                    logit = out.softmax(dim=1)
                    values, indice = ch.topk(logit, 2, dim=1)
                    logit_gap = values[:, 0] - values[:, 1]
                    logits.append({ "True_label": name, "logit_gap": logit_gap.item()})
                    index = index+1
            index = 0
    return logits

def evaluate_in(model, loaders, lr_tta=False):
    model.eval()
    obs = {'train_in':[],'train_out':[], 'test':[]}
    to_store = {'train_in':[],'train_out':[], 'test':[]}
    with ch.no_grad():
        for name in ['train_in','train_out', 'test']:
            total_correct, total_num = 0., 0.
            for ims, labs in tqdm(loaders[name]):
                with autocast():
                    out = model(ims)
                    logit = out.softmax(dim=1)
                    values, indice = ch.topk(logit, 2, dim=1)
                    logit_gap = values[:, 0] - values[:, 1]
                    obs[name].append(logit_gap.mean().cpu().item())
                    if lr_tta:
                        out += model(ims.flip(-1))
                    total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                    total_num += ims.shape[0]
            print(f'{name} accuracy: {total_correct / total_num * 100:.1f}%')
        print(f"in: {custom_norm_fit(obs['train_in'])}, out: {custom_norm_fit(obs['train_out'])},test: {custom_norm_fit(obs['test'])}")

def check_dataloaders_equal(loader1, loader2):
    # Check if the number of batches is the same
    if len(loader1) != len(loader2):
        return False

    for (data1, target1), (data2, target2) in zip(loader1, loader2):
        # Check if features are the same
        if not torch.equal(data1, data2):
            return False
        # Check if targets/labels are the same
        if not torch.equal(target1, target2):
            return False

    return True

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
'''
def check_dataloaders_equal(loader1, loader2):
    # Check if the number of batches is the same
    if len(loader1) != len(loader2):
        return False

    for (data1, target1), (data2, target2) in zip(loader1, loader2):
        # Check if features are the same
        if not ch.equal(data1, data2):
            return False
        # Check if targets/labels are the same
        if not ch.equal(target1, target2):
            return False

    return True
'''

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Loads from args.config_file if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    loaders,start_time = make_dataloaders()
    ensure_directory(config['data.results_folder'])

    model_filename = config['training.in_model_save_path']

    model_number = extract_integer_from_path(model_filename)

    if model_filename.endswith(('.pt', '.pth')):
        model = load_model(model_filename, f'cuda:{config["data.gpu"]}')

        out_index_file_path = config["data.out_dataset_index"]
        in_index_file_path = config["data.in_dataset_index"]

        logging.info(f"Processing model number: {model_number}")
        results_file_path = os.path.join(config['data.results_folder'], f'result_{model_number}.csv')


        logits = cal_p(model=model,in_dataset_index=in_index_file_path,out_dataset_index=out_index_file_path,loaders=loaders)
        pd.DataFrame(logits).to_csv(results_file_path, index=False)
