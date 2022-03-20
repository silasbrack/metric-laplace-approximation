import torch
from asdfghjkl.gradient import batch_gradient
from laplace import Laplace
from laplace.curvature.asdl import _get_batch_grad
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def jacobians(x, model, output_size=784):
    """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
    using asdfghjkl's gradient per output dimension.
    Parameters
    ----------
    x : torch.Tensor
        input data `(batch, input_shape)` on compatible device with model.
    Returns
    -------
    Js : torch.Tensor
        Jacobians `(batch, parameters, outputs)`
    f : torch.Tensor
        output function `(batch, outputs)`
    """
    Js = list()
    for i in range(output_size):
        def loss_fn(outputs, targets):
            return outputs[:, i].sum()

        f = batch_gradient(model, loss_fn, x, None).detach()
        Jk = _get_batch_grad(model)

        Js.append(Jk)
    Js = torch.stack(Js, dim=1)

    return Js, f


num_observations = 1000
num_outputs = 1

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
    nn.Linear(16, num_outputs)
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
    la.optimize_prior_precision()

    Js, f = jacobians(x, model, output_size=num_outputs)
    assert Js.shape == (num_observations, num_outputs, num_params)

    Hs = torch.einsum("nij,nkl->njl", Js, Js)
    assert Hs.shape == (num_observations, num_params, num_params)

    if hessian_structure == "diag":
        # Hs = torch.einsum("nij,nij->nj", Js, Js)
        Hs = torch.einsum("nii->ni", Hs)
        assert Hs.shape == (num_observations, num_params)

    # Hs.sum(dim=0), since we sum over the observations (that's what they seem to do)
    torch.testing.assert_close(la.H, Hs.sum(dim=0), rtol=1e-5, atol=0.)  # Less than 0.001% off
