from torchvision.datasets import SVHN
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class SVHNDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()

        self.name = "SVHN"
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4376821, 0.4437697, 0.47280442),
                    (0.19803012, 0.20101562, 0.19703614),
                ),
                # transforms.Grayscale(num_output_channels=1),
                # transforms.Resize(28),
            ]
        )

        self.df_train = None
        self.df_val = None
        self.df_test = None

    def prepare_data(self):
        # download
        SVHN(self.data_dir, split="train", download=True)
        SVHN(self.data_dir, split="test", download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train = SVHN(
                self.data_dir, split="train", transform=self.transform
            )
            self.df_train, self.df_val = random_split(train, [70000, 3257])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.df_test = SVHN(
                self.data_dir, split="test", transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.df_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.df_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.df_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )
