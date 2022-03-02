import datetime
import logging

import fire
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.lite import LightningLite
from pytorch_metric_learning import losses, miners
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models import ConvNet
from src.models.utils import test_model

logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(1234)


def run(epochs=10, lr=0.01, batch_size=64):
    model = ConvNet()
    loss_fn = losses.TripletMarginLoss()
    miner = miners.MultiSimilarityMiner()

    lite = Lite(gpus=1)
    lite.run(model,
             loss_fn,
             miner,
             epochs,
             lr,
             batch_size)

    lite.test()


def setup_logger():
    subdir = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    logdir = f"logs/{subdir}"
    writer = SummaryWriter(log_dir=logdir)
    return writer


class Lite(LightningLite):
    def run(self, model, loss_fn, miner, epochs, lr, batch_size):
        # LOGGING
        self.writer = setup_logger()

        # Data
        train_loader, val_loader, test_loader = self.setup_data(batch_size)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Lite setup
        model, optimizer = self.setup(model, optimizer)
        self.model = model

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

                self.writer.add_scalar("train_loss", loss, num_batches * epoch + i)

            average_train_loss = epoch_loss / num_batches
            self.writer.add_scalar("avg_train_loss", average_train_loss, epoch)

            # Validate
            self.validate(epoch)

        return model

    def validate(self, epoch):
        accuracy = test_model(self.train_loader.dataset,
                              self.val_loader.dataset,
                              self.model,
                              self.device)["precision_at_1"]

        self.writer.add_scalar("val_acc", accuracy, epoch)

        self.visualize(self.val_loader)

    def test(self):
        accuracy = test_model(self.train_loader.dataset,
                              self.val_loader.dataset,
                              self.model,
                              self.device)

        self.writer.add_scalar('test_acc', accuracy["precision_at_1"])

        self.visualize(self.test_loader)

    def visualize(self, dataloader):
        logging.info('Running TSNE')

        dataset = dataloader.dataset.dataset

        for image, target in dataloader:
            x = self.forward(image)

            embeddings = TSNE(learning_rate='auto', n_jobs=-1, init='random').fit_transform(x.cpu().detach().numpy())

            embeddings = embeddings

            for cls in dataset.classes:
                idx = target.cpu().detach().numpy() == dataset.class_to_idx[cls]
                plt.plot(embeddings[idx, 0], embeddings[idx, 1], marker='.', label=cls)

        self.writer.add_figure('T-SNE', plt.gcf())
        logging.info('Finished T-SNE')

    def setup_data(self, batch_size):
        # DATA
        data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=4, normalize=True)
        data.prepare_data()
        data.setup()

        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        test_loader = data.test_dataloader()

        return self.setup_dataloaders(train_loader, val_loader, test_loader)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    fire.Fire(run())
