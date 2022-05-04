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
from pl_bolts.datamodules import CIFAR10DataModule

from src.data.cifar100 import CIFAR100DataModule
from src.hessian.layerwise import ContrastiveHessianCalculator
from src.models.utils import test_model


class ConvDense(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
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

    latent_dim = 25
    net = ConvDense(latent_dim).to(device)
    # net = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(28*28*1, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 64),
    #     nn.ReLU(),
    #     nn.Linear(64, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, latent_dim),
    # ).to(device)
    num_params = sum(p.numel() for p in net.parameters())
    logging.info(f"Model has {num_params} parameters.")

    # data = CIFARData("data/", 16, 4)
    # data.setup()
    data = CIFAR10DataModule("data/", 16, 4)
    data.setup()
    loader = data.train_dataloader()

    # accuracy = test_model(
    #     loader.dataset, data.test_dataloader().dataset, net, device
    # )
    # logging.info(
    #     f"Accuracy before training is {100*accuracy['precision_at_1']:.2f}%."
    # )

    logging.info("Finding MAP solution.")
    epochs = 100
    lr = 3e-4
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

    accuracy = test_model(
        loader.dataset, data.test_dataloader().dataset, net, device
    )
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
        x_conv = net.conv(x).detach()
        output = net.linear(x_conv)
        # output = net(x)
        hard_pairs = miner(y)
        hessian = (
            compute_hessian(net.linear, output, x_conv, y, hard_pairs)
            / x.shape[0]
            * len(loader.dataset)
        )
        h.append(hessian)
        # h.append(compute_hessian(net, output, x, y, hard_pairs))
    h = torch.stack(h, dim=0).mean(dim=0).to(device)
    # print(f"{h=} before abs + 1")
    # print(f"{(h < 0).sum()} negative values")
    h = torch.abs(h) + 1
    # print(f"{h=}")

    mu_q = parameters_to_vector(net.linear.parameters())
    # mu_q = parameters_to_vector(net.parameters())
    sigma_q = 1 / (h + 1e-6)
    # print(f"{sigma_q=}")

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

    logging.info("Generating predictions from samples.")
    preds = generate_predictions_from_samples(data.test_dataloader(), samples, net, net.linear, device)
    preds = preds.detach().cpu()

    with open("preds.pkl", "wb") as f:
        pickle.dump(
            {"means": preds.mean(dim=0), "vars": preds.var(dim=0)},
            f,
        )

    logging.info("Generating ood predictions from samples.")

    ood_data = CIFAR100DataModule("data/", 16, 4)
    ood_data.setup()
    test_set = ood_data.dataset_test
    subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    mask = torch.tensor([test_set[i][1] in subset_classes for i in range(len(test_set))])
    indices = torch.arange(len(test_set))[mask]
    ood_dataset = Subset(test_set, indices)
    ood_dataloader = DataLoader(ood_dataset, batch_size=16, num_workers=4)
    preds_ood = generate_predictions_from_samples(ood_dataloader, samples, net, net.linear, device)
    preds_ood = preds_ood.detach().cpu()

    # preds_ood = generate_fake_predictions_from_samples(data.test_dataloader(), samples, net, net.linear, device)
    # preds_ood = preds_ood.detach().cpu()

    with open("preds_ood.pkl", "wb") as f:
        pickle.dump(
            {"means": preds_ood.mean(dim=0), "vars": preds_ood.var(dim=0)},
            f,
        )


def generate_predictions_from_samples(loader, weight_samples, full_model, inference_model=None, device="cpu"):
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


def generate_fake_predictions_from_samples(loader, weight_samples, full_model, inference_model=None, device="cpu"):
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


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
