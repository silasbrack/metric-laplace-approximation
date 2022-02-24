import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_metric_learning import distances, losses, miners, reducers, testers


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
        )

        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss = losses.TripletMarginLoss(
            margin=0.2, distance=self.distance, reducer=self.reducer
        )
        self.mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=self.distance, type_of_triplets="semihard"
        )

        self.evaluator = AccuracyCalculator(include=("precision_at_1",), k=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        embeddings = self(data)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss(embeddings, labels, indices_tuple)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        embeddings = self(data)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss(embeddings, labels, indices_tuple)

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        data, labels = batch
        embeddings = self(data)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss(embeddings, labels, indices_tuple)

        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
