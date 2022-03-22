import logging

import fire
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers
import torch
import torchmetrics
from laplace import Laplace
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule

from src.models.classification_conv_net import ConvNet


def run(epochs=20, lr=3e-4, batch_size=64, hessian='diag'):
    data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=4, normalize=True)
    data.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpus = 1 if torch.cuda.is_available() else 0
    model = ConvNet(lr=lr).to(device)

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=epochs,
        logger=loggers.TensorBoardLogger("logs/"),
    )
    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    trainer.test(dataloaders=data.test_dataloader())

    probs, labels, accuracy = predict(data.test_dataloader(), model, laplace=False)
    logging.info('[Softmax] Plotting')
    plot_calibration(labels, probs, accuracy, title='Softmax calibration curve')

    # Laplace post-hoc train
    la = Laplace(model, "classification",
                 subset_of_weights="last_layer",
                 hessian_structure=hessian)
    logging.info('[Laplace] Fitting Hessian')
    la.fit(data.train_dataloader())

    probs, labels, accuracy = predict(data.test_dataloader(), la, laplace=True)
    logging.info('[Laplace] Plotting')
    plot_calibration(labels, probs, accuracy, title=f'Laplace calibration curve ({hessian=})')

    # Laplace plotting
    logging.info('[Laplace] Optimizing')
    la.optimize_prior_precision(method='marglik', val_loader=data.val_dataloader())

    probs, labels, accuracy = predict(data.test_dataloader(), la, laplace=True)
    logging.info('[Laplace] Plotting')
    plot_calibration(labels, probs, accuracy, title=f'Optimized Laplace calibration curve ({hessian=})')


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []
    targets = []
    accuracy = torchmetrics.Accuracy()
    for x, y in dataloader:
        if laplace:
            pred = model(x, pred_type='glm', link_approx='probit')
            py.append(pred)
            targets.append(y)
            accuracy(pred, y)
        else:
            pred = torch.softmax(model(x), dim=-1)
            py.append(pred)
            targets.append(y)
            accuracy(pred, y)

    return torch.cat(py).cpu(), torch.cat(targets).cpu(), accuracy.compute()


def plot_calibration(labels, probs, accuracy, title='ECE calibration'):
    bins = 10
    accs, pred_probs, ece = calc_ece(probs, labels, bins)
    ece = ece.cpu().detach().numpy() * 100

    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

    ax[0].plot(pred_probs.numpy(), accs.numpy(), label='Actual')
    ax[0].plot(np.linspace(0, 1), np.linspace(0, 1), label='Identity')
    ax[0].set(
        xlim=[0, 1],
        ylim=[0, 1],
        ylabel='Accuracy',
        xlabel='Confidence',
        title=title
    )
    ax[0].text(x=0.1, y=0.75, s=f"{ece=:.2f}%\n{accuracy=:.2f}",
               bbox={'facecolor': 'blue', 'alpha': 0.25, 'pad': 5})

    ax[0].legend()

    conf, preds = torch.max(probs, dim=1)

    ax[1].hist(conf.numpy(), bins='scott')
    ax[1].set(
        xlabel="Confidence",
        ylabel="Frequency",
        title='Histogram of confidence'
    )

    fig.savefig(f'{title}.png')


def calc_ece(probs, labels, num_bins):
    maxp, predictions = probs.max(-1, keepdims=True)
    boundaries = torch.linspace(0, 1, num_bins + 1)
    lower_bound, upper_bound = boundaries[:-1], boundaries[1:]
    in_bin = maxp.ge(lower_bound).logical_and(maxp.lt(upper_bound)).float()
    bin_sizes = in_bin.sum(0)
    correct = predictions.eq(labels.unsqueeze(-1)).float()

    non_empty = bin_sizes.gt(0)
    accs = torch.where(non_empty, correct.mul(in_bin).sum(0) / bin_sizes, torch.zeros_like(bin_sizes))
    pred_probs = torch.where(non_empty, maxp.mul(in_bin).sum(0) / bin_sizes, torch.zeros_like(bin_sizes))
    bin_weight = bin_sizes / bin_sizes.sum()
    ece = accs.sub(pred_probs).abs().mul(bin_weight).sum()
    return accs, pred_probs, ece


if __name__ == "__main__":
    fire.Fire(run)
