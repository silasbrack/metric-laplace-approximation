import time
import logging

import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning.utils.loss_and_miner_utils import \
    get_all_pairs_indices
from torch.utils.data import DataLoader

from src.models.linear import LinearNet
from src.hessian import layerwise as lw
from src.hessian import rowwise as rw
from src.nearest_neighbors import get_nearest_latent_neighbors


def run():
    latent_size = 3
    model = LinearNet(latent_size).model

    data = CIFAR10DataModule("./data", batch_size=8, num_workers=0, normalize=True)
    data.setup()

    pair_dataset = get_nearest_latent_neighbors(data.dataset_test, model, latent_size, num_neighbors=5)
    pair_dataloader = DataLoader(pair_dataset, batch_size=32)

    t0 = time.perf_counter()
    Hs_row = rw.ContrastiveHessianCalculator().compute(pair_dataloader, model, latent_size)
    elapsed_row = time.perf_counter() - t0
    print(Hs_row)

    t0 = time.perf_counter()
    Hs_layer = lw.ContrastiveHessianCalculator().compute(pair_dataloader, model, latent_size)
    elapsed_layer = time.perf_counter() - t0

    logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")
    torch.testing.assert_close(Hs_layer, Hs_row, rtol=1e-2, atol=0.)  # Less than 1% off

    # for x, labels in data.train_dataloader():
    #     a1, p, a2, n = get_all_pairs_indices(labels, labels)
    #     x1 = x[torch.cat((a1, a2))]
    #     x2 = x[torch.cat((p, n))]
    #     y = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))
    #
    #     t0 = time.perf_counter()
    #     Hs_row = rw.ContrastiveHessianCalculator().compute_batch(model, latent_size, x1, x2, y)
    #     elapsed_row = time.perf_counter() - t0
    #
    #     t0 = time.perf_counter()
    #     Hs_layer = lw.ContrastiveHessianCalculator().compute_batch(model, latent_size, x1, x2, y)
    #     elapsed_layer = time.perf_counter() - t0
    #
    #     logging.info(f"{elapsed_row=}")
    #     logging.info(f"{elapsed_layer=}")
    #     torch.testing.assert_close(Hs_layer, Hs_row, rtol=1e-2, atol=0.)  # Less than 1% off
    #     break


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
