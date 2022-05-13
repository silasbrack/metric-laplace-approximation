import logging
import pickle
import torch
from torch.utils.data import Subset, DataLoader
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)

from src.post_hoc_laplace import ConvDense
from src.data import CIFAR100DataModule, SVHNDataModule
from src.train_metric import train_metric


RESULTS_FOLDER = "./results/ensemble"


def train_deep_ensemble(id_, ood, latent_dim=25, num_ensembles=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if id_ == "fashionmnist":
        ensembles = [
            ConvDense(1, latent_dim).to(device) for _ in range(num_ensembles)
        ]
        data = FashionMNISTDataModule("data/", 16, 4)
        data.prepare_data()
        data.setup()
    elif id_ == "cifar10":
        ensembles = [
            ConvDense(3, latent_dim).to(device) for _ in range(num_ensembles)
        ]
        data = CIFAR10DataModule("data/", 16, 4)
        data.setup()
    else:
        raise NotImplementedError

    ensembles = [
        train_metric(ensemble, data.train_dataloader(), device)
        for ensemble in ensembles
    ]

    logging.info("Generating predictions from samples.")
    preds = generate_predictions_from_ensemble(
        data.test_dataloader(), ensembles, device
    )
    preds = preds.detach().cpu()

    with open(
        f"{RESULTS_FOLDER}/preds_ensemble_{ood}_{latent_dim}.pkl", "wb"
    ) as f:
        pickle.dump(
            {"means": preds.mean(dim=0), "vars": preds.var(dim=0)},
            f,
        )

    logging.info("Generating ood predictions from samples.")

    if ood == "cifar100":
        ood_data = CIFAR100DataModule("data/", 16, 4)
        ood_data.setup()
        test_set = ood_data.dataset_test
        subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mask = torch.tensor(
            [test_set[i][1] in subset_classes for i in range(len(test_set))]
        )
        indices = torch.arange(len(test_set))[mask]
        ood_dataset = Subset(test_set, indices)
        ood_dataloader = DataLoader(ood_dataset, batch_size=16, num_workers=4)
        preds_ood = generate_predictions_from_ensemble(
            ood_dataloader, ensembles, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "mnist":
        ood_data = MNISTDataModule("data/", 16, 4)
        ood_data.setup()
        preds_ood = generate_predictions_from_ensemble(
            ood_data.test_dataloader(), ensembles, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "svhn":
        ood_data = SVHNDataModule("data/", 16, 4)
        ood_data.prepare_data()
        ood_data.setup()
        preds_ood = generate_predictions_from_ensemble(
            ood_data.test_dataloader(), ensembles, device
        )
        preds_ood = preds_ood.detach().cpu()
    else:
        raise NotImplementedError

    with open(
        f"{RESULTS_FOLDER}/preds_ood_ensemble_{ood}_{latent_dim}.pkl", "wb"
    ) as f:
        pickle.dump(
            {"means": preds_ood.mean(dim=0), "vars": preds_ood.var(dim=0)},
            f,
        )


def generate_predictions_from_ensemble(loader, ensembles, device="cpu"):
    preds = []
    for ensemble in ensembles:
        ensemble_preds = []
        for x, _ in iter(loader):
            x = x.to(device)
            pred = ensemble(x)
            ensemble_preds.append(pred)
        preds.append(torch.cat(ensemble_preds, dim=0))
    return torch.stack(preds, dim=0)


if __name__ == "__main__":
    # train_deep_ensemble("fashionmnist", "mnist", latent_dim=2)
    # train_deep_ensemble("cifar10", "svhn", latent_dim=2)
    # train_deep_ensemble("cifar10", "svhn", latent_dim=25)
    # train_deep_ensemble("cifar10", "cifar100", latent_dim=2)
    train_deep_ensemble("cifar10", "cifar100", latent_dim=25)
