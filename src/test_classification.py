import logging
import time

import torch
from laplace import Laplace
from torch import nn

from src.data.cifar import CIFARData
from src.hessian.backpack import HessianCalculator


def run():
    data = CIFARData("./data", batch_size=32, num_workers=0)
    data.setup()
    loader = data.test_dataloader()

    # model = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(28 * 28 * 1, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 10),
    # )
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(676, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )
    lossfunc = nn.CrossEntropyLoss()

    #  layers_to_estimate=model[-5:] for all linear layers
    calculator = HessianCalculator(
        model,
        lossfunc,
        layers_to_estimate=model[-1],  # Last layer
    )
    t0 = time.perf_counter()
    Hs_backpack = calculator.compute(loader)
    elapsed_backpack = time.perf_counter() - t0

    la = Laplace(
        model,
        "classification",
        hessian_structure="diag",
        subset_of_weights="last_layer",
    )
    t0 = time.perf_counter()
    la.fit(loader)
    Hs_la = la.H
    elapsed_la = time.perf_counter() - t0

    logging.info(f"{elapsed_la=}")
    logging.info(f"{elapsed_backpack=}")

    torch.testing.assert_close(Hs_la, Hs_backpack, rtol=1e-3, atol=0.0)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
