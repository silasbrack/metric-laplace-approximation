import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_metric_learning import distances, losses, miners, reducers


class ConvNet(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
        )

        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss = losses.TripletMarginLoss(
            margin=0.2, distance=self.distance, reducer=self.reducer
        )
        self.mining_func = miners.TripletMarginMiner(
            margin=0.2, distance=self.distance, type_of_triplets="semihard"
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        embeddings = self(data)
        indices_tuple = self.mining_func(embeddings, labels)
        loss = self.loss(embeddings, labels, indices_tuple)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
