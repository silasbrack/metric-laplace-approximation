import fire
import torch
from laplace import Laplace
from pl_bolts.datamodules import CIFAR10DataModule
from torch import nn

from src.models.train_helper import test_model
from src.models.train_metric_learning import train


def run(epochs=10, lr=0.01, batch_size=64):

    data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=4, normalize=True)
    data.setup()

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Conv2d(3, 10, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(500, 50),
    ).to(device)

    model = train(device, epochs, lr, model, train_loader, val_loader)

    la = Laplace(model, "metric",
                 subset_of_weights="last_layer",
                 hessian_structure="diag")
    la.fit(data.train_dataloader())
    la.eval = lambda: None
    # la.optimize_prior_precision(method='CV', val_loader=data.val_dataloader())
    accuracy = test_model(data.dataset_train, data.dataset_test, la, "cpu")["precision_at_1"]


if __name__ == "__main__":
    fire.Fire(run())

