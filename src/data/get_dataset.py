import os
from pl_bolts.datamodules import CIFAR10DataModule

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_dataset():
    datamodule = CIFAR10DataModule(dir_path + '\cifar10')
    datamodule.setup()
    return datamodule
