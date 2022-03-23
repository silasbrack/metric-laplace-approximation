import torch
from asdfghjkl.gradient import batch_gradient
from laplace.curvature.asdl import _get_batch_grad
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning import miners
from torch import nn


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


class SiameseModel(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(6272, latent_size),
        )

    def forward(self, x):
        x1 = x[:, 0, :]
        x2 = x[:, 1, :]
        z1 = self.model(x1)
        z2 = self.model(x2)
        return torch.cat((z1, z2), dim=1)


class SingleModel(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(6272, latent_size),
        )

    def forward(self, x):
        return self.model(x)


batch_size = 128
# input_size = 1
latent_size = 3

# For contrastive loss, we give one pair at a time, so we have 2 outputs from our model, or 2 inputs to our loss.
num_outputs = 2

model = SiameseModel(latent_size)
singlemodel = SingleModel(latent_size)
num_params = sum(p.numel() for p in model.parameters())

miner = miners.MultiSimilarityMiner()

data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=0, normalize=True)
data.setup()

x, y = next(iter(data.train_dataloader()))
X = torch.stack((x, x), dim=1)
y = torch.randint(low=0, high=2, size=(batch_size,))

Js, f = jacobians(X, model, output_size=num_outputs*latent_size)
assert Js.shape == (batch_size, num_outputs*latent_size, num_params)
assert f.shape == (batch_size, num_outputs*latent_size)

Jz1 = Js[:, :3, :]
Jz2 = Js[:, 3:, :]

Hs = torch.einsum("nij,nij->nj", Jz1, Jz1) + \
     torch.einsum("nij,nij->nj", Jz2, Jz2) - \
     2 * (
         torch.einsum("nij,nij->nj", Jz1, Jz2) +
         torch.einsum("nij,nij->nj", Jz2, Jz1)
     )
assert Hs.shape == (batch_size, num_params)

Jz1, _ = jacobians(x, singlemodel, output_size=latent_size)
Jz2, _ = jacobians(x, singlemodel, output_size=latent_size)

Hs_new = torch.einsum("nij,nij->nj", Jz1, Jz1) + \
     torch.einsum("nij,nij->nj", Jz2, Jz2) - \
     2 * (
         torch.einsum("nij,nij->nj", Jz1, Jz2) +
         torch.einsum("nij,nij->nj", Jz2, Jz1)
     )
assert Hs_new.shape == (batch_size, num_params)

print()

# # FULL RANK
# # Hs = Jz1.T @ Jz1 + Jz2.T @ Jz2 - 2 * (Jz1.T @ Jz2 + Jz2.T @ Jz1)
# Hs = torch.einsum("nij,nkl->njl", Jz1, Jz1) + \
#      torch.einsum("nij,nkl->njl", Jz2, Jz2) - \
#      2 * (
#          torch.einsum("nij,nkl->njl", Jz1, Jz2) +
#          torch.einsum("nij,nkl->njl", Jz2, Jz1)
#      )

# L = y * ||z_1 - z_2||^2 + (1 - y) max(0, m - ||z_1 - z_2||^2)
# The Hessian is equal to what we calculated, except when we have:
# 1. A negative pair
# 2. The margin minus the norm is negative
m = 0.1  # margin
mask = torch.logical_and(
    (1-y).bool(),
    m - torch.norm(f, dim=1) < 0
)
# We want to set each [num_params x num_params] Hessian to 0 for these observations
mask = mask.view(-1, 1).expand(batch_size, num_params)
Hs = Hs.masked_fill_(mask, 0.)

Hs = Hs.sum(dim=0)

prior_precision = torch.ones((num_params,))  # Prior precision is one.
precision = Hs + prior_precision
covariance_matrix = 1 / precision
assert covariance_matrix.shape == (num_params,)
