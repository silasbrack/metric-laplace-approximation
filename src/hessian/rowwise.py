import torch
from asdfghjkl import batch_gradient
from laplace.curvature.asdl import _get_batch_grad


def jacobians(x, model, output_size=784):
    """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\)
       at current parameter \\(\\theta\\)
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


def compute_hessian_rmse(loader, model, output_size, hessian_structure="diag", agg="sum"):
    temp = []
    for x, _ in loader:
        Hs = compute_hessian_rmse_batch(hessian_structure, model, output_size, x)
        temp.append(Hs)
    Hs = torch.cat(temp)
    if agg == "sum":
        Hs = Hs.sum(dim=0)
    return Hs


def compute_hessian_rmse_batch(hessian_structure, model, output_size, x):
    Js, f = jacobians(x, model, output_size=output_size)
    if hessian_structure == "diag":
        Hs = torch.einsum("nij,nij->nj", Js, Js)
    elif hessian_structure == "full":
        Hs = torch.einsum("nij,nkl->njl", Js, Js)
    else:
        raise NotImplementedError
    return Hs


def compute_hessian_contrastive_batch(x1, x2, y, model, output_size, margin=0.1, hessian_structure="diag", agg="sum"):
    Jz1, f1 = jacobians(x1, model, output_size)
    Jz2, f2 = jacobians(x2, model, output_size)

    # L = y * ||z_1 - z_2||^2 + (1 - y) max(0, m - ||z_1 - z_2||^2)
    # The Hessian is equal to Hs, except when we have:
    # 1. A negative pair
    # 2. The margin minus the norm is negative
    mask = torch.logical_and(
        (1 - y).bool(),
        margin - torch.einsum("no,no->n", f1 - f2, f1 - f2) < 0
        # margin - torch.pow(torch.linalg.norm(f1 - f2, dim=1), 2) < 0
    )
    if hessian_structure == "diag":
        Hs = torch.einsum("nij,nij->nj", Jz1, Jz1) + \
             torch.einsum("nij,nij->nj", Jz2, Jz2) - \
             2 * (
                     torch.einsum("nij,nij->nj", Jz1, Jz2) +
                     torch.einsum("nij,nij->nj", Jz2, Jz1)
             )
        mask = mask.view(-1, 1).expand(*Hs.shape)
    elif hessian_structure == "full":
        Hs = torch.einsum("nij,nkl->njl", Jz1, Jz1) + \
             torch.einsum("nij,nkl->njl", Jz2, Jz2) - \
             2 * (
                     torch.einsum("nij,nkl->njl", Jz1, Jz2) +
                     torch.einsum("nij,nkl->njl", Jz2, Jz1)
             )
        mask = mask.view(-1, 1, 1).expand(*Hs.shape)
    else:
        raise NotImplementedError

    Hs = Hs.masked_fill_(mask, 0.)

    if agg == "sum":
        Hs = Hs.sum(dim=0)

    return Hs
