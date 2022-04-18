import logging

import torch
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Adam

from src.data.cifar import CIFARData
from src.hessian.rowwise import ContrastiveHessianCalculator


def compute_kl_term(mu_q, sigma_q):
    k = len(mu_q)
    kl = 0.5 * (
            torch.log(1.0 / sigma_q)
            - k
            + torch.dot(mu_q, mu_q)
            + torch.sum(sigma_q)
    )
    return kl


def sample_neural_network_wights(parameters, posterior_scale, n_samples=32):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


def run():
    contrastive_loss = losses.ContrastiveLoss()
    miner = miners.MultiSimilarityMiner()
    compute_hessian = ContrastiveHessianCalculator().compute_batch_pairs

    latent_dim = 10
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28*1, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, latent_dim),
    )
    num_params = sum(p.numel() for p in net.parameters())

    optim = Adam(net.parameters(), lr=3e-4)

    data = CIFARData("data/", 16, 4)
    data.setup()
    loader = data.train_dataloader()

    epochs = 10
    h = 1e10 * torch.ones((num_params,))
    alpha = 1e-4  # try to play a bit with this
    for epoch in range(epochs):
        epoch_losses = []
        for x, y in loader:
            optim.zero_grad()

            mu_q = parameters_to_vector(net.parameters())
            sigma_q = 1 / (h + 1e-6)

            kl = compute_kl_term(mu_q, sigma_q)
            sampled_nn = sample_neural_network_wights(mu_q, sigma_q)

            con_loss = []
            h = []
            for nn_i in sampled_nn:
                vector_to_parameters(nn_i, net.parameters())

                output = net(x)

                hard_pairs = miner(output, y)
                con_loss.append(contrastive_loss(output, y, hard_pairs))
                h.append(compute_hessian(net, output, x, y, hard_pairs))
            con_loss = torch.cat(con_loss).mean()
            h = torch.cat(h).mean()
            loss = con_loss + alpha * kl
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
        loss = torch.mean(torch.tensor(epoch_losses))
        logging.info(f"{loss=} for {epoch=}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
