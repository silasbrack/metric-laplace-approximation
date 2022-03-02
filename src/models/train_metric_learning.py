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
from MulticoreTSNE import MulticoreTSNE as TSNE
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import umap
import umap.plot

from src.models import ConvNet
from src.models.utils import test_model

plt.switch_backend('agg')
logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(1234)


def run(epochs=1000, lr=0.01, batch_size=64):
    model = ConvNet()
    loss_fn = losses.TripletMarginLoss()
    miner = miners.TripletMarginMiner()

    params = {
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'model': model.__class__.__name__,
        'miner': miner.__class__.__name__,
        'loss_fn': loss_fn.__class__.__name__,
        'cuda': torch.cuda.is_available()
    }

    logging.info(f'Parameters: {params}')

    lite = Lite(gpus=1 if torch.cuda.is_available() else 0)

    lite.run(name='metric-triplet-miner',
             model=model,
             loss_fn=loss_fn,
             miner=miner,
             epochs=epochs,
             lr=lr,
             batch_size=batch_size,
             freq=2,
             load_dir=None
             )

    lite.test()


def setup_logger(name):
    subdir = get_time()
    logdir = f"logs/{name}/{subdir}"
    writer = SummaryWriter(log_dir=logdir)
    return writer


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')


class Lite(LightningLite):
    # noinspection PyMethodOverriding
    def run(self, name, model, loss_fn, miner, epochs, lr, batch_size, freq, load_dir=None):
        start_time = get_time()

        # LOGGING
        self.name = name
        self.writer = setup_logger(name)

        # Data
        train_loader, val_loader, test_loader = self.setup_data(batch_size)
        self.train_loader, self.val_loader, self.test_loader = train_loader, val_loader, test_loader

        # Load model
        if load_dir:
            state_dict = self.load(load_dir)
            model.load_state_dict(state_dict)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Lite setup
        model, optimizer = self.setup(model, optimizer)
        self.model = model

        num_batches = len(train_loader)
        for epoch in range(epochs):
            self.epoch = epoch
            logging.info(f"Epoch: {epoch}")
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

            # Validate frequency
            if (epoch != 0) and (epoch % freq == 0):
                logging.info('Validating model')
                self.validate()

                filepath = f"models/{name}/{start_time}/checkpoint_{epoch}.ckpt"
                logging.info(f'Saving model @ {filepath}')
                self.save(
                    content=model.module.state_dict(),
                    filepath=filepath
                )

        return model

    def validate(self):
        accuracy = test_model(self.train_loader.dataset,
                              self.val_loader.dataset,
                              self.model,
                              self.device)

        self.writer.add_scalar("val_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar("val_map", accuracy['mean_average_precision'], self.epoch)

        self.visualize(self.val_loader, self.val_loader.dataset.dataset.class_to_idx)

    def test(self):
        accuracy = test_model(self.train_loader.dataset,
                              self.test_loader.dataset,
                              self.model,
                              self.device)

        self.writer.add_scalar("test_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar("test_map", accuracy['mean_average_precision'], self.epoch)

        self.visualize(self.test_loader, self.test_loader.dataset.class_to_idx)

    def visualize(self, dataloader, class_to_idx):
        logging.info(f'Running visualization, epoch {self.epoch}')

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        images = []
        targets = []

        for image, target in dataloader:
            images.append(image)
            targets.append(target)

        image = torch.cat(images, dim=0)
        target = torch.cat(targets, dim=0)

        x = self.forward(image)

        target = target.cpu().detach().numpy()
        x = x.cpu().detach().numpy()

        idx_to_class = {v: k for k, v in class_to_idx.items()}

        labels = np.array([idx_to_class[i] for i in target])

        mapper = umap.UMAP().fit(x)
        umap.plot.points(mapper, labels=labels, ax=ax)

        self.writer.add_figure('UMAP', fig, self.epoch)

        logging.info('Finished UMAP')

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
    fire.Fire(run)
