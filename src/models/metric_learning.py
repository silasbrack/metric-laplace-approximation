import datetime
import logging

import numpy as np
import torch
import umap
import umap.plot
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.lite import LightningLite
from torch.utils.tensorboard import SummaryWriter

from src.models.utils import test_model

plt.switch_backend('agg')
logging.getLogger().setLevel(logging.INFO)
torch.manual_seed(1234)


def setup_logger(name):
    subdir = get_time()
    logdir = f"logs/{name}/{subdir}"
    writer = SummaryWriter(log_dir=logdir)
    return writer


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')


class Lite(LightningLite):
    def init(self, name, model, loss_fn, miner, batch_size, optimizer, load_dir):
        # LOGGING
        self.name = name
        self.writer = setup_logger(name)

        # Data
        self.batch_size = batch_size
        self.train_loader, self.val_loader, self.test_loader = self.setup_data(batch_size)

        # Load model
        if load_dir:
            state_dict = self.load(load_dir)
            model.load_state_dict(state_dict)

        # Miners and Loss
        self.loss_fn = loss_fn
        self.miner = miner

        # Lite setup
        self.model, self.optimizer = self.setup(model, optimizer)
        self.epoch = 0

    def train(self, epochs, freq):
        return self.run(epochs, freq)

    # noinspection PyMethodOverriding
    def run(self, epochs, freq):
        logging.info(f'Training')
        self.model.train()

        start_time = get_time()

        if not self.name:
            raise ValueError('Please run .init()')

        for epoch in range(epochs):
            self.epoch = epoch

            logging.info(f"Epoch: {epoch}")
            epoch_loss = 0.0
            for i, (image, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(image)

                hard_pairs = self.miner(output, target)
                loss = self.loss_fn(output, target, hard_pairs)
                self.backward(loss)
                epoch_loss += loss.item()
                self.optimizer.step()

            average_train_loss = epoch_loss / (i + 1)
            self.writer.add_scalar("train_loss", average_train_loss, global_step=epoch, new_style=True)

            # Validate @ frequency
            if epoch % freq == 0:
                self.validate()

                filepath = f"models/{self.name}/{start_time}/checkpoint_{epoch}.ckpt"
                logging.info(f'Saving model @ {filepath}')
                self.save(
                    content=self.model.module.state_dict(),
                    filepath=filepath
                )

        logging.info(f'Finished training @ epoch: {self.epoch}')
        self.log_hyperparams()
        return self.model

    def validate(self):
        logging.info(f'Validating @ epoch: {self.epoch}')

        self.model.eval()
        val_loss = 0.0
        for i, (image, target) in enumerate(self.val_loader):
            output = self.model(image)

            hard_pairs = self.miner(output, target)
            loss = self.loss_fn(output, target, hard_pairs)
            val_loss += loss.item()

        average_val_loss = val_loss / (i + 1)
        self.writer.add_scalar("val_loss", average_val_loss, global_step=self.epoch, new_style=True)

        accuracy = test_model(self.train_loader.dataset,
                              self.val_loader.dataset,
                              self.model,
                              self.device)

        self.writer.add_scalar("val_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar("val_map", accuracy['mean_average_precision'], self.epoch)

        self.visualize(self.val_loader, self.val_loader.dataset.dataset.class_to_idx)

    def test(self):
        logging.info(f'Testing @ epoch: {self.epoch}')
        accuracy = test_model(self.train_loader.dataset,
                              self.test_loader.dataset,
                              self.model,
                              self.device)

        self.writer.add_scalar("test_acc", accuracy["precision_at_1"], self.epoch)
        self.writer.add_scalar("test_map", accuracy['mean_average_precision'], self.epoch)

        self.visualize(self.test_loader, self.test_loader.dataset.class_to_idx)

    def visualize(self, dataloader, class_to_idx):
        logging.info(f'Visualizing @ epoch: {self.epoch}')

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

    def log_hyperparams(self):
        logging.info('Logging hyperparameters')

        logging.info('Calculating train accuracy')
        train_accuracy = test_model(self.train_loader.dataset,
                                    self.train_loader.dataset,
                                    self.model,
                                    self.device)

        logging.info('Calculating validation accuracy')
        val_accuracy = test_model(self.train_loader.dataset,
                                  self.val_loader.dataset,
                                  self.model,
                                  self.device)

        logging.info('Calculating test accuracy')
        test_accuracy = test_model(self.train_loader.dataset,
                                   self.test_loader.dataset,
                                   self.model,
                                   self.device)

        self.writer.add_hparams(
            hparam_dict={
                'name': self.name,
                'miner': self.miner.__class__.__name__,
                'loss_fn': self.loss_fn.__class__.__name__,
                'epoch': self.epoch + 1,
                'lr': self.optimizer.defaults['lr'],
                'batch_size': self.batch_size,
                'model': self.model.module.__class__.__name__,
            },
            metric_dict={
                'train_acc': train_accuracy['precision_at_1'],
                'train_map': train_accuracy['mean_average_precision'],
                'val_acc': val_accuracy['precision_at_1'],
                'val_map': val_accuracy['mean_average_precision'],
                'test_acc': test_accuracy['precision_at_1'],
                'test_map': test_accuracy['mean_average_precision']
            },
            run_name=".")
