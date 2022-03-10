import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torchmetrics


class ConvNet(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(6272, 128),
            nn.Linear(128, 10),
        )
        # self.model = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(3*32*32, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(64, 10),
        # )
        self.loss = CrossEntropyLoss()
        self.test_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(data)
        loss = self.loss(output, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        output = self(data)
        loss = self.loss(output, labels)
        self.val_accuracy(output, labels)

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, labels = batch
        output = self(data)
        loss = self.loss(output, labels)
        self.test_accuracy(output, labels)

        self.log("test_loss", loss, prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log('validation_accuracy', self.val_accuracy, prog_bar=True)

    def test_epoch_end(self, outs):
        self.log('test_accuracy', self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
