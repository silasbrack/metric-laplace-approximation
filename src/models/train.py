import logging
import warnings

from src.models.MetricLite import MetricLite

warnings.filterwarnings('ignore')

import fire
import torch
from pytorch_metric_learning import losses, miners
from torch import optim

from src.models import ConvNet
from src.models.MetricLaplace import MetricLaplace


def run(loss='contrastive',
        miner='multisimilarity',
        epochs=0,
        freq=2,
        lr=3e-4,
        batch_size=64,
        name='testing'):

    model = ConvNet()

    if loss == 'contrastive':
        loss_fn = losses.ContrastiveLoss()
    elif loss == 'triplet':
        loss_fn = losses.TripletMarginLoss()
    else:
        raise ValueError(f"{loss} is not a recognized loss")

    if miner == 'multisimilarity':
        miner_fn = miners.MultiSimilarityMiner()
    elif miner == 'triplet':
        miner_fn = miners.TripletMarginMiner()
    else:
        raise ValueError(f"{miner} is not a recognized miner")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    params = {
        'name': name,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'model': model.__class__.__name__,
        'miner': miner_fn.__class__.__name__,
        'loss_fn': loss_fn.__class__.__name__,
        'cuda': torch.cuda.is_available()
    }

    logging.info(f'Parameters: {params}')

    lite = MetricLite(gpus=1 if torch.cuda.is_available() else 0)

    # Init
    lite.init(name=name,
              model=model,
              loss_fn=loss_fn,
              miner=miner_fn,
              batch_size=batch_size,
              optimizer=optimizer,
              load_dir=None
              )

    # Training loop
    # lite.train(epochs=epochs,
    #            freq=freq,
    #            )

    # Testing
    # lite.test()

    # Log hyperparams
    # lite.log_hyperparams()

    # Laplace
    la = MetricLaplace(lite)

    la.train_laplace()


if __name__ == "__main__":
    fire.Fire(run)
