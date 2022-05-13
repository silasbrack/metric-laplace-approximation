from abc import abstractmethod

import torch
from asdfghjkl import batch_gradient
from laplace.curvature.asdl import _get_batch_grad


class HessianCalculator:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def compute_batch(self, *args, **kwargs):
        pass

    def compute(self, loader, model, output_size):
        hessian = None
        for batch in loader:

            batch = [item.to(self.device) for item in batch]

            Hs = self.compute_batch(model, output_size, *batch)
            if hessian is None:
                hessian = Hs
            else:
                hessian += Hs
        return hessian


class RmseHessianCalculator(HessianCalculator):
    def __init__(self, hessian_structure):
        self.hessian_structure = hessian_structure

    def compute_batch(self, model, output_size, x, *args, **kwargs):
        Js, f = jacobians(x, model, output_size=output_size)
        if self.hessian_structure == "diag":
            Hs = torch.einsum("nij,nij->nj", Js, Js)
        elif self.hessian_structure == "full":
            Hs = torch.einsum("nij,nkl->njl", Js, Js)
        else:
            raise NotImplementedError
        return Hs.sum(0)


class ContrastiveHessianCalculator(HessianCalculator):
    def __init__(self, margin=0.2, hessian="diag"):
        self.margin = margin
        self.hessian = hessian

    def compute_batch(self, model, output_size, x1, x2, y, *args, **kwargs):
        Jz1, f1 = jacobians(x1, model, output_size)
        Jz2, f2 = jacobians(x2, model, output_size)

        # L = y * ||z_1 - z_2||^2 + (1 - y) max(0, m - ||z_1 - z_2||^2)
        # The Hessian is equal to Hs, except when we have:
        # 1. A negative pair
        # 2. The margin minus the norm is negative
        mask = torch.logical_and(
            (1 - y).bool(),
            self.margin - torch.einsum("no,no->n", f1 - f2, f1 - f2) < 0
            # margin - torch.pow(torch.linalg.norm(f1 - f2, dim=1), 2) < 0
        )
        if self.hessian == "diag":
            Hs = (
                torch.einsum("nij,nij->nj", Jz1, Jz1)
                + torch.einsum("nij,nij->nj", Jz2, Jz2)
                - 2
                * (
                    torch.einsum("nij,nij->nj", Jz1, Jz2)
                    + torch.einsum("nij,nij->nj", Jz2, Jz1)
                )
            )
            mask = mask.view(-1, 1).expand(*Hs.shape)
        elif self.hessian == "full":
            Hs = (
                torch.einsum("nij,nkl->njl", Jz1, Jz1)
                + torch.einsum("nij,nkl->njl", Jz2, Jz2)
                - 2
                * (
                    torch.einsum("nij,nkl->njl", Jz1, Jz2)
                    + torch.einsum("nij,nkl->njl", Jz2, Jz1)
                )
            )
            mask = mask.view(-1, 1, 1).expand(*Hs.shape)
        else:
            raise NotImplementedError

        Hs = Hs.masked_fill_(mask, 0.0)

        return Hs.sum(dim=0)

    def compute_batch_pairs(self, model, embeddings, x, target, hard_pairs):
        ap, p, an, n = hard_pairs

        x1 = x[torch.cat((ap, an))]
        x2 = x[torch.cat((p, n))]

        t = torch.cat(
            (
                torch.ones(p.shape[0], device=x.device),
                torch.zeros(n.shape[0], device=x.device),
            )
        )

        return self.compute_batch(model, embeddings.shape[-1], x1, x2, t)


def jacobians(x, model, output_size=784):
    """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\)
       at current parameter \\(\\theta\\)
    using asdfghjkl's gradient per output dimension.
    Parameters
    ----------
    x : torch.Tensor
        input data `(batch, input_shape)` on compatible device with
        model.
    Returns
    -------
    Js : torch.Tensor
        Jacobians `(batch, parameters, outputs)`
    f : torch.Tensor
        output function `(batch, outputs)`
    """
    jacobians = list()
    f = None
    for i in range(output_size):

        def loss_fn(outputs, targets):
            return outputs[:, i].sum()

        f = batch_gradient(model, loss_fn, x, None).detach()
        jacobian_i = _get_batch_grad(model)

        jacobians.append(jacobian_i)
    jacobians = torch.stack(jacobians, dim=1)
    return jacobians, f
