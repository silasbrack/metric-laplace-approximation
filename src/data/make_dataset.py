# -*- coding: utf-8 -*-
from pl_bolts.datamodules import CIFAR10DataModule


def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    cifar10 = CIFAR10DataModule(input_filepath)
    cifar10.prepare_data()
    cifar10.setup()


if __name__ == "__main__":
    main("cifar10")
