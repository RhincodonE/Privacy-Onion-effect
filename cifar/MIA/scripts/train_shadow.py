"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.

First, from the same directory, run:

    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`

to generate the FFCV-formatted versions of CIFAR.

Then, simply run this to train models with default hyperparameters:

    `python train_cifar.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_cifar.py --config-file default_config.yaml \
                           --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""
from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
import os
import torch as ch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss, Conv2d, BatchNorm2d
from torch.optim import SGD, lr_scheduler
import torchvision

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

Section('training', 'Hyperparameters').params(
    lr=Param(float, 'The learning rate to use', required=True),
    epochs=Param(int, 'Number of epochs to run for', required=True),
    lr_peak_epoch=Param(int, 'Peak epoch for cyclic lr', required=True),
    batch_size=Param(int, 'Batch size', default=512),
    momentum=Param(float, 'Momentum for SGD', default=0.9),
    weight_decay=Param(float, 'l2 weight decay', default=5e-4),
    label_smoothing=Param(float, 'Value of label smoothing', default=0.1),
    num_workers=Param(int, 'The number of workers', default=8),
    lr_tta=Param(bool, 'Test time augmentation by averaging with horizontally flipped version', default=True),
    in_model_save_path=Param(str, 'model save addr', default=True),
)

Section('data', 'data related stuff').params(
    in_dataset=Param(str, 'addr to store in dataset', required=True),
    out_dataset=Param(str, 'addr to store out dataset', required=True),
    val_dataset=Param(str, 'addr to store val dataset', required=True),
    model_folder=Param(str, 'folder to store the models', required=True),
    gpu=Param(int, 'GPU to use', required=True),
)
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

@param('data.in_dataset')
@param('data.out_dataset')
@param('data.val_dataset')
@param('training.batch_size')
@param('training.num_workers')
@param('data.gpu')
def make_dataloaders(in_dataset=None, out_dataset=None, val_dataset=None, batch_size=None, num_workers=None, gpu=0):
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
        if name == 'train_in':
            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])
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




    return loaders, start_time

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
    model = model.to(memory_format=ch.channels_last).cuda(int(gpu))
    return model

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

@param('training.lr')
@param('training.epochs')
@param('training.momentum')
@param('training.weight_decay')
@param('training.label_smoothing')
@param('training.lr_peak_epoch')
def train_in(model, loaders, lr=None, epochs=None, label_smoothing=None,
          momentum=None, weight_decay=None, lr_peak_epoch=None):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loaders['train_in'])
    # Cyclic LR with single triangle
    lr_schedule = np.interp(np.arange((epochs+1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

    for _ in range(epochs):
        for ims, labs in tqdm(loaders['train_in']):
            opt.zero_grad(set_to_none=True)
            with autocast():
                out = model(ims)
                loss = loss_fn(out, labs)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()


@param('training.lr_tta')
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
        
if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    # Loads from args.config_file if provided
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()

    loaders, start_time = make_dataloaders()

    model_in = construct_model()
    print(check_dataloaders_equal(loaders['train_in'],loaders['train_out']))
    train_in(model_in, loaders)
    evaluate_in(model_in, loaders)
    print(f'Total time: {time.time() - start_time:.5f}')

    if not os.path.exists(config['data.model_folder']):
        os.makedirs(config['data.model_folder'])
        print(f"Created directory: {config['data.model_folder']}")
    else:
        print(f"Directory already exists: {config['data.model_folder']}")

    ch.save(model_in.state_dict(), config['training.in_model_save_path'])
