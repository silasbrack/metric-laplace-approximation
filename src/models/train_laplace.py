import fire
import pytorch_lightning as pl
import torch
import torchmetrics
from laplace import Laplace
from matplotlib import pyplot as plt
from pl_bolts.datamodules import CIFAR10DataModule

from src.models.classification_conv_net import ConvNet


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []
    targets = []
    accuracy = torchmetrics.Accuracy()
    for x, y in dataloader:
        if laplace:
            pred = model(x)
            py.append(pred)
            targets.append(y)
            accuracy(pred, y)
        else:
            pred = torch.softmax(model(x), dim=-1)
            py.append(pred)
            targets.append(y)
            accuracy(pred, y)

    return torch.cat(py).cpu(), torch.cat(targets).cpu(), accuracy.compute()


def run(epochs=1, lr=3e-4, batch_size=64):

    data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=4, normalize=True)
    data.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet(lr=lr).to(device)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
    )
    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
    trainer.test(dataloaders=data.test_dataloader())

    la = Laplace(model, "classification",
                 subset_of_weights="last_layer",
                 hessian_structure="diag")
    la.fit(data.train_dataloader())
    # la.optimize_prior_precision(method='CV', val_loader=data.val_dataloader())
    # accuracy = test_model(data.dataset_train, data.dataset_test, la, "cpu")["precision_at_1"]
    probs, labels, accuracy = predict(data.test_dataloader(), la, laplace=True)
    accs, pred_probs, ece = calc_ece(probs, labels, 10)
    print(accs, pred_probs, ece)
    fig, ax = plt.subplots()
    ax.plot(pred_probs.numpy(), accs.numpy())
    ax.plot()
    print(accuracy)


def calc_ece(probs, labels, num_bins):
    maxp, predictions = probs.max(-1, keepdims=True)
    boundaries = torch.linspace(0, 1, num_bins+1)
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
    fire.Fire(run())

