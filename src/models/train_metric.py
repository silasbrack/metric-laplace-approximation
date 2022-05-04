import logging
import warnings

from src.models.metric_lite import MetricLite

warnings.filterwarnings('ignore')

import fire
import torch
from pytorch_metric_learning import losses, miners
from torch import optim, nn

from src.models import ConvNet, LinearNet


def run(
    loss="contrastive",
    miner="multisimilarity",
    model="conv",
    epochs=5,
    freq=2,
    lr=3e-4,
    batch_size=64,
    name="testing",
):

    if model == "conv":
        model = ConvNet()
    elif model == "linear":
        model = LinearNet()
    else:
        raise ValueError(f"{model} is not a recognized model")

    if loss == "contrastive":
        loss_fn = losses.ContrastiveLoss()
    elif loss == "triplet":
        loss_fn = losses.TripletMarginLoss()
    else:
        raise ValueError(f"{loss} is not a recognized loss")

    if miner == "multisimilarity":
        miner_fn = miners.MultiSimilarityMiner()
    elif miner == "triplet":
        miner_fn = miners.TripletMarginMiner(type_of_triplets="hard")
    else:
        raise ValueError(f"{miner} is not a recognized miner")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    params = {
        "name": name,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "model": model.__class__.__name__,
        "miner": miner_fn.__class__.__name__,
        "loss_fn": loss_fn.__class__.__name__,
        "cuda": torch.cuda.is_available()
    }

    logging.info(f"Parameters: {params}")

    lite = MetricLite(gpus=1 if torch.cuda.is_available() else 0)

    # Init
    lite.init(
        name=name,
        model=model,
        loss_fn=loss_fn,
        miner=miner_fn,
        batch_size=batch_size,
        optimizer=optimizer,
        load_dir=None,
        to_visualize=True
    )

    # Training loop
    lite.train(epochs=epochs,
               freq=freq)

    # Testing
    lite.test()

    # Log hyperparams
    lite.log_hyperparams()

    # return lite


if __name__ == "__main__":
    fire.Fire(run)
