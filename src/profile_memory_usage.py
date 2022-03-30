import time

import torch
from memory_profiler import memory_usage
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.manual_hessian import RmseHessianCalculator


def compute_hessian(x, feature_maps, net, output_size, h_scale):
    H = []
    bs = x.shape[0]
    feature_maps = [x] + feature_maps
    tmp = torch.diag_embed(torch.ones(bs, output_size, device=x.device))

    with torch.no_grad():
        for k in range(len(net) - 1, -1, -1):

            # compute Jacobian wrt input
            if isinstance(net[k], torch.nn.Linear):
                diag_elements = torch.diagonal(tmp, dim1=1, dim2=2)
                feature_map_k2 = (feature_maps[k] ** 2).unsqueeze(1)

                h_k = torch.bmm(diag_elements.unsqueeze(2), feature_map_k2).view(bs, -1)

                # has a bias
                if net[k].bias is not None:
                    h_k = torch.cat([h_k, diag_elements], dim=1)

                H = [h_k] + H

            # compute Jacobian wrt input
            if isinstance(net[k], torch.nn.Tanh):
                J_tanh = torch.diag_embed(
                    torch.ones(feature_maps[k + 1].shape, device=x.device)
                    - feature_maps[k + 1] ** 2
                )
                tmp = torch.einsum("bnm,bnj,bjk->bmk", J_tanh, tmp, J_tanh)
            elif isinstance(net[k], torch.nn.ReLU):
                J_relu = torch.diag_embed(
                    (feature_maps[k] > 0).float()
                )
                tmp = torch.einsum("bnm,bnj,bjk->bmk", J_relu, tmp, J_relu)

            if k == 0:
                break

            # compute Jacobian wrt weight
            if isinstance(net[k], torch.nn.Linear):
                tmp = torch.einsum("nm,bnj,jk->bmk",
                                   net[k].weight,
                                   tmp,
                                   net[k].weight)

    H = torch.cat(H, dim=1)

    # mean over batch size scaled by the size of the dataset
    H = h_scale * torch.mean(H, dim=0)

    return H


num_observations = 1000
X = torch.rand((num_observations, 1)).float()
y = 4.5 * torch.cos(2 * torch.pi * X + 1.5 * torch.pi) - \
    3 * torch.sin(4.3 * torch.pi * X + 0.3 * torch.pi) + \
    3.0 * X - 7.5
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=num_observations)
x, y = next(iter(dataloader))

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)


def row_wise_proc():
    RmseHessianCalculator().calculate_hessian(
        x,
        model=model,
        num_outputs=1,
        hessian_structure="diag",
    )


activation = []
def get_activation():
    def hook(model, input, output):
        activation.append(output.detach())
    return hook
for layer in model:
    layer.register_forward_hook(get_activation())
output = model(x)


def layer_wise_proc():
    compute_hessian(x, activation, model, 1, num_observations)

t0 = time.perf_counter()
mem_row_wise = memory_usage(
    # proc=row_wise_proc,
    proc=(
        RmseHessianCalculator().calculate_hessian,
        [x],
        dict(model=model, num_outputs=1, hessian_structure="diag")
    ),
    # max_usage=True,
    # timestamps=True,
)
wall_row_wise = time.perf_counter() - t0

t0 = time.perf_counter()
mem_layer_wise = memory_usage(
    # proc=layer_wise_proc,
    proc=(
        compute_hessian,
        [x, activation, model, 1, num_observations]
    ),
    # max_usage=True,
    # timestamps=True,
)
wall_layer_wise = time.perf_counter() - t0

print()
