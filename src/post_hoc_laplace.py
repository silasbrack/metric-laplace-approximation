import logging
import pickle
from tqdm import tqdm

import torch
from torch import nn
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
)
from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Adam
from pl_bolts.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
import wandb

from src.data import CIFAR100DataModule, SVHNDataModule
from src.hessian.layerwise import ContrastiveHessianCalculator
from src.models.utils import test_model


class ConvDense(nn.Module):
    def __init__(self, channels, latent_dim):
        super().__init__()
        if channels == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
            )
            self.linear = nn.Sequential(
                nn.Linear(4608, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
        elif channels == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
            )
            self.linear = nn.Sequential(
                nn.Linear(6272, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x


class AllPermutationsMiner:
    def __call__(self, y):
        batch_size = y.shape[0]
        indices = torch.arange(batch_size)
        pairs = torch.combinations(indices)
        mask = y[pairs[:, 0]] == y[pairs[:, 1]]
        a1, p = pairs[mask, 0], pairs[mask, 1]
        a2, n = pairs[~mask, 0], pairs[~mask, 1]
        return (a1, p, a2, n)


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # id_ = "fashionmnist"
    # ood = "mnist"
    id_ = "cifar10"
    ood = "svhn"
    epochs = 20
    lr = 3e-4
    batch_size = 16

    latent_dim = 2
    if id_ == "fashionmnist":
        net = ConvDense(1, latent_dim).to(device)
        data = FashionMNISTDataModule("data/", batch_size, 4)
        data.prepare_data()
        data.setup()
    elif id_ == "cifar10":
        net = ConvDense(3, latent_dim).to(device)
        data = CIFAR10DataModule("data/", batch_size, 4)
        data.setup()
    else:
        raise NotImplementedError
    # num_params = sum(p.numel() for p in net.parameters())
    # logging.info(f"Model has {num_params} parameters.")

    wandb.init(
        project="metric-laplace-approximation",
        name="post-hoc-laplace",
        config={
            "learning_rate": lr,
            "epochs": epochs,
            "model": "ConvDense",
            "latent_dim": latent_dim,
            "miner": "MultiSimilarityMiner",
            "loss": "ContrastiveLoss",
            "optimizer": "Adam",
            "id": id_,
            "ood": ood,
        },
    )
    wandb.watch(net)

    loader = data.train_dataloader()

    # accuracy = test_model(
    #     loader.dataset, data.test_dataloader().dataset, net, device
    # )
    # logging.info(
    #     f"Accuracy before training is {100*accuracy['precision_at_1']:.2f}%."
    # )

    logging.info("Finding MAP solution.")
    miner = miners.MultiSimilarityMiner()
    contrastive_loss = losses.ContrastiveLoss()
    optim = Adam(net.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            output = net(x)
            hard_pairs = miner(output, y)
            loss = contrastive_loss(output, y, hard_pairs)
            loss.backward()
            optim.step()
            wandb.log({"Training loss": loss.item(), "Epoch": epoch})

    accuracy = test_model(
        loader.dataset, data.test_dataloader().dataset, net, device
    )
    wandb.summary["Post hoc accuracy"] = accuracy["precision_at_1"]
    logging.info(
        f"Accuracy after training is {100*accuracy['precision_at_1']:.2f}%."
    )

    logging.info("Computing hessian.")
    calculator = ContrastiveHessianCalculator()
    calculator.init_model(net.linear)
    compute_hessian = calculator.compute_batch_pairs
    miner = AllPermutationsMiner()
    h = []
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        x_conv = net.conv(x)  # .detach()
        output = net.linear(x_conv)
        # output = net(x)
        hard_pairs = miner(y)
        hessian = (
            compute_hessian(net.linear, output, x_conv, y, hard_pairs)
            / x.shape[0]
            * len(loader.dataset)
        )
        h.append(hessian)
    h = torch.stack(h, dim=0).sum(dim=0).to(device)
    if (h < 0).sum():
        logging.warn("Found negative values in Hessian.")
    h += 1

    mu_q = parameters_to_vector(net.linear.parameters())
    # mu_q = parameters_to_vector(net.parameters())
    sigma_q = 1 / (h + 1e-6)

    wandb.log({"Hessian": h})
    wandb.log({"Covariance": sigma_q})

    def sample(parameters, posterior_scale, n_samples=16):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device=parameters.device)
        samples = samples * posterior_scale.reshape(1, n_params)
        return parameters.reshape(1, n_params) + samples

    logging.info("Sampling.")
    samples = sample(mu_q, sigma_q)

    device = "cpu"
    net = net.to(device)
    samples = samples.to(device)

    accuracies = get_sample_accuracy(
        loader.dataset,
        data.test_dataloader().dataset,
        net,
        net.linear,
        samples,
        device,
    )
    logging.info(f"Sample accuracies = {accuracies}")
    # wandb.log({"Sample accuracies", accuracies})

    logging.info("Generating predictions from samples.")
    preds = generate_predictions_from_samples(
        data.test_dataloader(), samples, net, net.linear, device
    )
    preds = preds.detach().cpu()

    wandb.log(
        {
            "Prediction mean": preds.mean(dim=0),
            "Prediction variance": preds.var(dim=0),
        }
    )
    with open("preds.pkl", "wb") as f:
        pickle.dump(
            {"means": preds.mean(dim=0), "vars": preds.var(dim=0)},
            f,
        )

    logging.info("Generating ood predictions from samples.")

    if ood == "cifar100":
        ood_data = CIFAR100DataModule("data/", batch_size, 4)
        ood_data.setup()
        test_set = ood_data.dataset_test
        subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mask = torch.tensor(
            [test_set[i][1] in subset_classes for i in range(len(test_set))]
        )
        indices = torch.arange(len(test_set))[mask]
        ood_dataset = Subset(test_set, indices)
        ood_dataloader = DataLoader(
            ood_dataset, batch_size=batch_size, num_workers=4
        )
        preds_ood = generate_predictions_from_samples(
            ood_dataloader, samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "mnist":
        ood_data = MNISTDataModule("data/", batch_size, 4)
        ood_data.setup()
        preds_ood = generate_predictions_from_samples(
            ood_data.test_dataloader(), samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "svhn":
        ood_data = SVHNDataModule("data/", batch_size, 4)
        ood_data.prepare_data()
        ood_data.setup()
        preds_ood = generate_predictions_from_samples(
            ood_data.test_dataloader(), samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "noise":
        preds_ood = generate_fake_predictions_from_samples(
            data.test_dataloader(), samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    else:
        raise NotImplementedError

    wandb.log(
        {
            "OOD prediction mean": preds_ood.mean(dim=0),
            "OOD prediction variance": preds_ood.var(dim=0),
        }
    )
    with open("preds_ood.pkl", "wb") as f:
        pickle.dump(
            {"means": preds_ood.mean(dim=0), "vars": preds_ood.var(dim=0)},
            f,
        )

    wandb.finish()


def generate_predictions_from_samples(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for x, _ in iter(loader):
            x = x.to(device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


def generate_fake_predictions_from_samples(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for x, _ in iter(loader):
            x = torch.randn(x.shape, device=device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


def get_sample_accuracy(
    train_set, test_set, model, inference_model, samples, device
):
    accuracies = []
    for sample in samples:
        vector_to_parameters(sample, inference_model.parameters())
        accuracies.append(
            test_model(train_set, test_set, model, device)["precision_at_1"]
        )
    return accuracies


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
