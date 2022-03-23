import torch
from laplace import Laplace
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.manual_hessian import RmseHessianCalculator

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
    nn.Linear(16, 1)
)
num_params = sum(p.numel() for p in model.parameters())

for hessian_structure in ["diag", "full"]:
    la = Laplace(
        model,
        "regression",
        hessian_structure=hessian_structure,
        subset_of_weights="all",
    )
    la.fit(dataloader)

    Hs = RmseHessianCalculator().calculate_hessian(
        x,
        model=model,
        num_outputs=1,
        hessian_structure=hessian_structure,
    )

    torch.testing.assert_close(la.H, Hs, rtol=1e-5, atol=0.)  # Less than 0.001% off

    # Prior precision is one.
    if hessian_structure == "diag":
        prior_precision = torch.ones((num_params,))  # Prior precision is one.
        precision = Hs + prior_precision
        covariance_matrix = 1 / precision
        torch.testing.assert_close(la.posterior_variance, covariance_matrix, rtol=1e-5, atol=0.)
    elif hessian_structure == "full":
        prior_precision = torch.eye(num_params)
        precision = Hs + prior_precision
        covariance_matrix = torch.inverse(precision)
        torch.testing.assert_close(la.posterior_covariance, covariance_matrix, rtol=1e-1, atol=0.)
    else:
        raise NotImplementedError
