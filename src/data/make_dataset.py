# -*- coding: utf-8 -*-
import logging
import os

import torch
from torchvision import datasets, transforms
from pl_bolts.datamodules import CIFAR10DataModule

def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    cifar10 = CIFAR10DataModule(input_filepath)
    cifar10.prepare_data()
    print(next(cifar10.train_dataloader()))

if __name__ == "__main__":
    main("cifar10")
    
