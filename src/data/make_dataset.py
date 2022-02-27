from pl_bolts.datamodules import CIFAR10DataModule


def main():
    """
    Downloads data to /data folder.
    """
    CIFAR10DataModule("./data").prepare_data()


if __name__ == "__main__":
    main()
