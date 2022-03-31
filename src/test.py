from builtins import breakpoint
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys

from src.models.manual_hessian import RmseHessianCalculator

sys.path.append("../../Laplace")
from laplace.laplace import Laplace
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def create_dataset():

    N = 1000
    X = np.random.rand(N)
    y = (
        4.5 * np.cos(2 * np.pi * X + 1.5 * np.pi)
        - 3 * np.sin(4.3 * np.pi * X + 0.3 * np.pi)
        + 3.0 * X
        - 7.5
    )
    X = torch.tensor(X).unsqueeze(1).type(torch.float)
    y = torch.tensor(y).type(torch.float).unsqueeze(-1)

    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, pin_memory=True)

    return dataloader


def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 30),
        torch.nn.Tanh(),
        torch.nn.Linear(30, 20),
        torch.nn.Tanh(),
        torch.nn.Linear(20, 10),
        torch.nn.Tanh(),
        torch.nn.Linear(10, 1),
    )


def train_model(dataset, model):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(1):
        total, correct = 0, 0
        for X, y in dataset:

            optimizer.zero_grad()

            yhat = model(X)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()

def compute_hessian_laplace_redux(model, dataloader):

    la = Laplace(
        model,
        "regression",
        hessian_structure="diag",
        subset_of_weights="all",
    )

    la.fit(dataloader)

    la.optimize_prior_precision()

    return la.H.numpy()

def compute_hessian_ours(dataloader, net):
    output_size = 1

    # keep track of running sum
    H_running_sum = torch.zeros_like(parameters_to_vector(net.parameters()))
    counter = 0

    feature_maps = []

    def fw_hook_get_latent(module, input, output):
        feature_maps.append(output.detach())

    for k in range(len(net)):
        net[k].register_forward_hook(fw_hook_get_latent)

    for x, y in dataloader:

        feature_maps = []
        yhat = net(x)

        bs = x.shape[0]
        feature_maps = [x] + feature_maps
        tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

        H = []
        with torch.no_grad():
            for k in range(len(net) - 1, -1, -1):
                if isinstance(net[k], torch.nn.Linear):
                    diag_elements = torch.diagonal(tmp, dim1=1, dim2=2)
                    feature_map_k2 = (feature_maps[k] ** 2).unsqueeze(1)

                    h_k = torch.bmm(diag_elements.unsqueeze(2), feature_map_k2).view(
                        bs, -1
                    )

                    # has a bias
                    if net[k].bias is not None:
                        h_k = torch.cat([h_k, diag_elements], dim=1)

                    H = [h_k] + H

                elif isinstance(net[k], torch.nn.Tanh):
                    J_tanh = torch.diag_embed(
                        torch.ones(feature_maps[k + 1].shape, device=x.device)
                        - feature_maps[k + 1] ** 2
                    )
                    # TODO: make more efficent by using row vectors
                    tmp = torch.einsum("bnm,bnj,bjk->bmk", J_tanh, tmp, J_tanh)

                if k == 0:
                    break

                if isinstance(net[k], torch.nn.Linear):
                    tmp = torch.einsum(
                        "nm,bnj,jk->bmk", net[k].weight, tmp, net[k].weight
                    )

            counter += len(torch.cat(H, dim=1))
            H_running_sum += torch.cat(H, dim=1).sum(0)

    assert counter == dataloader.dataset.__len__()

    final_H = H_running_sum

    return final_H


if __name__ == "__main__":

    train = True
    laplace_redux = True
    laplace_ours = True
    laplace_row = True

    dataset = create_dataset()
    model = create_model()

    if train:
        train_model(dataset, model)

    model.eval()

    if laplace_redux:
        t0 = time.perf_counter()
        H = compute_hessian_laplace_redux(model, dataset)
        wall = time.perf_counter() - t0
        breakpoint()

    if laplace_ours:
        t0 = time.perf_counter()
        H_layer = compute_hessian_ours(dataset, model)
        wall_layer = time.perf_counter() - t0
        breakpoint()

    if laplace_row:
        calculator = RmseHessianCalculator()

        t0 = time.perf_counter()
        H_row = calculator.calculate_hessian(
            dataset,
            model=model,
            num_outputs=1,
            hessian_structure="diag",
        )
        wall_row = time.perf_counter() - t0
        breakpoint()

    breakpoint()
    err = (H - H_layer.numpy()) / H
    plt.plot(err, "-o")
    plt.show()
