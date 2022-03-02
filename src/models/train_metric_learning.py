import logging

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.lite import LightningLite
from pytorch_metric_learning import losses, miners
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.train_helper import test_model

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(1234)


def run(epochs=10, lr=0.01, batch_size=64):

    data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=4, normalize=True)
    data.setup()

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Conv2d(3, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(500, 50),
    )

    model = Lite(gpus=1).run(epochs, lr, model, train_loader, val_loader)
    accuracy = test_model(data.dataset_train, data.dataset_test, model, device)["precision_at_1"]
    print(f"Accuracy = {100*accuracy:.2f}")


class Lite(LightningLite):
    def run(self, epochs, lr, model, train_loader, val_loader):
        writer = SummaryWriter(log_dir="logs/")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = losses.TripletMarginLoss()
        miner = miners.MultiSimilarityMiner()

        model, optimizer = self.setup(model, optimizer)
        train_loader, val_loader = self.setup_dataloaders(train_loader, val_loader)

        num_batches = len(train_loader)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i, (image, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(image)
                hard_pairs = miner(output, target)
                loss = loss_fn(output, target, hard_pairs)
                self.backward(loss)
                epoch_loss += loss.item()
                optimizer.step()
                writer.add_scalar("train_loss", loss, num_batches * epoch + i)

            average_epoch_loss = epoch_loss / num_batches
            writer.add_scalar("epoch_loss", average_epoch_loss, epoch)
            accuracy = test_model(train_loader.dataset, val_loader.dataset, model, self.device)["precision_at_1"]
            writer.add_scalar("val_acc", accuracy, epoch)
        return model


if __name__ == "__main__":
    fire.Fire(run())
