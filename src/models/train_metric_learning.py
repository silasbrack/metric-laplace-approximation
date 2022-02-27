import logging

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pl_bolts import datamodules as dm
from pytorch_metric_learning import losses, miners
from tqdm import tqdm

from src.models.train_helper import test_model

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(1234)


class Net(nn.Module):
    """
    Network architecture.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        return self.fc1(x)


def run(epochs=20, lr=0.01, batch_size=64):
    writer = SummaryWriter(log_dir="logs/")

    data = dm.CIFAR10DataModule("./data", batch_size=batch_size, num_workers=4, normalize=True)
    data.setup()

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_fn = losses.TripletMarginLoss()
    miner = miners.MultiSimilarityMiner()

    num_batches = len(train_loader)
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_bar = tqdm(train_loader)
        epoch_bar.set_description(f"Epoch {epoch}")
        for i, (image, target) in enumerate(epoch_bar):
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(image)
            hard_pairs = miner(output, target)
            loss = loss_fn(output, target, hard_pairs)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            writer.add_scalar("train_loss", loss, num_batches*epoch + i)
            epoch_bar.set_postfix({"Loss": loss, "Pos pairs": miner.num_pos_pairs, "Neg pairs": miner.num_neg_pairs})
        average_epoch_loss = epoch_loss / num_batches
        writer.add_scalar("epoch_loss", average_epoch_loss, epoch)
        accuracy = test_model(train_loader.dataset, val_loader.dataset, model, device)["precision_at_1"]
        writer.add_scalar("val_acc", accuracy, epoch)


if __name__ == "__main__":
    fire.Fire(run())
