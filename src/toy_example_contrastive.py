import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.cifar import CIFARData
from src.hessian import layerwise as lw
from src.hessian import rowwise as rw
from src.nearest_neighbors import get_nearest_latent_neighbors


def run():
    latent_size = 3
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    data = CIFARData("./data", batch_size=8, num_workers=0)
    data.setup()

    pair_dataset = get_nearest_latent_neighbors(data.dataset_test, model, latent_size, num_neighbors=5)
    pair_dataloader = DataLoader(pair_dataset, batch_size=32)

    t0 = time.perf_counter()
    Hs_row = rw.ContrastiveHessianCalculator().compute(pair_dataloader, model, latent_size)
    elapsed_row = time.perf_counter() - t0
    logging.info(f"{elapsed_row=}")

    t0 = time.perf_counter()
    Hs_layer = lw.ContrastiveHessianCalculator().compute(pair_dataloader, model, latent_size)
    elapsed_layer = time.perf_counter() - t0
    logging.info(f"{elapsed_layer=}")

    torch.testing.assert_close(Hs_layer, Hs_row, rtol=1e-2, atol=0.)  # Less than 1% off


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
