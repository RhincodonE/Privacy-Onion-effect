from argparse import ArgumentParser
from typing import List
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset

import torch as ch
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField
import time

Section('data', 'arguments to give the writer').params(
    in_dataset=Param(str, 'Where to write the in dataset', required=True),
    val_dataset=Param(str, 'Where to write the val dataset', required=True),
    out_dataset=Param(str, 'Where to write the in dataset', required=True),
    out_dataset_index=Param(str, 'Where to write the in dataset', required=True),
    in_dataset_index=Param(str, 'Where to write the in dataset', required=True),
)


@param('data.in_dataset')
@param('data.val_dataset')
@param('data.out_dataset')
@param('data.out_dataset_index')
@param('data.in_dataset_index')
def main(in_dataset, val_dataset, out_dataset, out_dataset_index,in_dataset_index,test = True):
# Set seed for reproducibility
    np.random.seed(int(time.time()))

# Load the full CIFAR10 training dataset
    full_train_dataset = torchvision.datasets.CIFAR10('./data/tmp_shadow', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10('./data/tmp_shadow', train=False, download=True)

    indices = np.random.permutation(len(full_train_dataset))
    midpoint = len(indices) // 2
    in_indices = indices[midpoint:]
    out_indices = indices[:midpoint]
    np.save(out_dataset_index, out_indices)
    np.save(in_dataset_index, in_indices)

# Create subsets for 'in' and 'out'
    in_subset = Subset(full_train_dataset, in_indices)
    out_subset = Subset(full_train_dataset, out_indices)

    datasets = {
        'in': in_subset,
        'test': test_dataset,
        'out': out_subset
        }

    for (name, ds) in datasets.items():
        if name == 'in':
            path = in_dataset
        elif name == 'out':
            path = out_dataset
        else:
            path = val_dataset

        writer = DatasetWriter(path, {
            'image': RGBImageField(),
            'label': IntField()
        })

        writer.from_indexed_dataset(ds)


if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()
