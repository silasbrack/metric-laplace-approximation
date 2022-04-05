import time
import logging

import torch
from laplace import Laplace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.hessian import layerwise as lw
from src.hessian import rowwise as rw


def run():
    num_observations = 1000
    output_size = 1

    torch.manual_seed(42)

    X = torch.rand((num_observations, output_size)).float()
    y = 4.5 * torch.cos(2 * torch.pi * X + 1.5 * torch.pi) - \
        3 * torch.sin(4.3 * torch.pi * X + 0.3 * torch.pi) + \
        3.0 * X - 7.5
    y = y[:, 0].squeeze()
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)

    model = nn.Sequential(
        nn.Linear(1, 16),
        nn.Tanh(),
        nn.Linear(16, 32),
        nn.Tanh(),
        nn.Linear(32, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 16),
        nn.Tanh(),
        nn.Linear(16, 1)
    )
    hessian_structure = "diag"
    la = Laplace(
        model,
        "regression",
        hessian_structure=hessian_structure,
        subset_of_weights="all",
    )
    t0 = time.perf_counter()
    la.fit(dataloader)
    elapsed_la = time.perf_counter() - t0

    t0 = time.perf_counter()
    Hs_row = rw.RmseHessianCalculator(hessian_structure).compute(dataloader, model, output_size)
    elapsed_row = time.perf_counter() - t0

    t0 = time.perf_counter()
    Hs_layer = lw.RmseHessianCalculator().compute(dataloader, model, output_size)
    elapsed_layer = time.perf_counter() - t0

    logging.info(f"{elapsed_la=}")
    logging.info(f"{elapsed_row=}")
    logging.info(f"{elapsed_layer=}")

    torch.testing.assert_close(la.H, Hs_row, rtol=1e-3, atol=0.)  # Less than 0.01% off
    torch.testing.assert_close(la.H, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off
    torch.testing.assert_close(Hs_row, Hs_layer, rtol=1e-3, atol=0.)  # Less than 0.01% off


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()

# from torch.distributions import Normal
# from torch.nn.utils import parameters_to_vector, vector_to_parameters
# mu_q = parameters_to_vector(model.parameters())
#
# def sample(parameters, posterior_scale, n_samples=100):
#     n_params = len(parameters)
#     samples = torch.randn(n_samples, n_params, device="cpu")
#     samples = samples * posterior_scale.reshape(1, n_params)
#     return parameters.reshape(1, n_params) + samples
#
# sigma_q = 1 / (Hs + 1e-6)
#
# preds = []
# samples = sample(mu_q, sigma_q, n_samples=100)
# for net_sample in samples:
#     vector_to_parameters(net_sample, model.parameters())
#     batch_preds = []
#     for x, _ in dataloader:
#         pred = model(x)
#         batch_preds.append(pred)
#     preds.append(torch.cat(batch_preds, dim=0))
# preds = torch.stack(preds)
# means = preds.mean(dim=0)
# vars = preds.var(dim=0)


# # Prior precision is one.
# num_params = sum(p.numel() for p in model.parameters())
# if hessian_structure == "diag":
#     prior_precision = torch.ones((num_params,))  # Prior precision is one.
#     precision = Hs + prior_precision
#     covariance_matrix = 1 / precision
#     torch.testing.assert_close(la.posterior_variance, covariance_matrix, rtol=1e-5, atol=0.)
# elif hessian_structure == "full":
#     prior_precision = torch.eye(num_params)
#     precision = Hs + prior_precision
#     covariance_matrix = torch.inverse(precision)
#     torch.testing.assert_close(la.posterior_covariance, covariance_matrix, rtol=1e-1, atol=0.)
# else:
#     raise NotImplementedError
