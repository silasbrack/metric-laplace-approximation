# -*- coding: utf-8 -*-
import logging
import os

import torch
from torchvision import datasets, transforms
from pl_bolts import CIFAR10DataModule

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    cifar10 = CIFAR10DataModule(input_filepath)

if __name__ == "__main__":
    main("cifar10")
    
