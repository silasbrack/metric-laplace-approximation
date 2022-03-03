import logging

import fire
import torch
from pytorch_metric_learning import losses, miners
from torch import optim

from src.models import ConvNet
from src.models.metric_learning import Lite


def run(epochs=6,
        freq=2,
        lr=3e-4,
        batch_size=64,
        name='testing'):

    model = ConvNet()
    loss_fn = losses.TripletMarginLoss()
    miner = miners.MultiSimilarityMiner()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    params = {
        'name': name,
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

    # Init
    lite.init(name=name,
              model=model,
              loss_fn=loss_fn,
              miner=miner,
              batch_size=batch_size,
              optimizer=optimizer,
              load_dir=None
              )

    # Training loop
    lite.train(epochs=epochs,
               freq=freq,
               )

    # Testing
    lite.test()


if __name__ == "__main__":
    fire.Fire(run)
