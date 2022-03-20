import datetime
import logging

import numpy as np
import torch
import umap
import umap.plot
from laplace import Laplace
from laplace.curvature import BackPackGGN
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.lite import LightningLite
from torch.nn.modules.loss import _Loss
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


# class MyLoss:
#     def __init__(self, miner, loss_fn):
#         self.miner = miner
#         self.loss_fn = loss_fn


class MyBackend(BackPackGGN):
    def __init__(self, model, likelihood, backend_kwargs, **kwargs):
        super().__init__(model, likelihood, **kwargs)

        def calc_loss(X, y):
            hard_pairs = backend_kwargs['miner'](X, y)
            loss = backend_kwargs['loss_fn'](X, y, hard_pairs)
            return loss

        self.lossfunc = calc_loss
        self.factor = 1


class MetricLaplace:
    def __init__(self, lite):
        # Laplace
        self.la = Laplace(lite.model, "regression",
                          subset_of_weights="last_layer",
                          hessian_structure='kron',
                          backend=MyBackend,
                          backend_kwargs={
                              'miner': lite.miner,
                              'loss_fn': lite.loss_fn
                          })

        self.lite = lite

        return

    def train_laplace(self):
        logging.info('Laplace Training')
        self.la.fit(self.lite.train_loader)
        logging.info('Laplace Optimizing')
        self.la.optimize_prior_precision(method='marglik', val_loader=self.lite.val_loader)

    # def predict_laplace(self):
    #     x = X_test.flatten().cpu().numpy()
    #     f_mu, f_var = la(X_test)
    #     f_mu = f_mu.squeeze().detach().cpu().numpy()
    #     f_sigma = f_var.squeeze().sqrt().cpu().numpy()
    #     pred_std = np.sqrt(f_sigma ** 2 + la.sigma_noise.item() ** 2)
