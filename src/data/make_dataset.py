# -*- coding: utf-8 -*-
from pl_bolts.datamodules import CIFAR10DataModule
import argparse


def main(output_path='cifar10'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    cifar10 = CIFAR10DataModule(output_path)
    cifar10.prepare_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output_path', type=str, help='File path from current working directory to install the cifar10 dataset')

    args = parser.parse_args()

    main(args.output_path)
