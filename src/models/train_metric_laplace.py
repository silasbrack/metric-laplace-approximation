


import logging

import fire
import torch
from laplace import Laplace
from pl_bolts.datamodules import CIFAR10DataModule

from src.models import MetricLite
from src.models.manual_hessian import calculate_hessian
from src.models.train_metric import run as train_metric


def run(batch_size=64, hessian='diag'):
    data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=0, normalize=True)
    data.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 1 if torch.cuda.is_available() else 0
    model: MetricLite = train_metric()

    # Laplace post-hoc train
    la = Laplace(model, "regression",
                 subset_of_weights="last_layer",
                 hessian_structure=hessian)
    logging.info('[Laplace] Fitting Hessian Manually')
    la.H = calculate_hessian(x, y, model, model.model.latent_dim, loss="contrastive")


if __name__ == "__main__":
    fire.Fire(run)
