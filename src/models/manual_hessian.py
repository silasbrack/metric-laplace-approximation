import torch
from asdfghjkl.gradient import batch_gradient
from laplace.curvature.asdl import _get_batch_grad


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


def calculate_hessian(x, y, model, num_outputs, hessian_structure="diag", loss="rmse", agg="sum"):
    Js, f = jacobians(x, model, output_size=num_outputs)

    if loss == "rmse":
        if hessian_structure == "diag":
            Hs = torch.einsum("nij,nij->nj", Js, Js)
        elif hessian_structure == "full":
            Hs = torch.einsum("nij,nkl->njl", Js, Js)
        else:
            raise NotImplementedError
    elif loss == "contrastive":
        Jz1 = Js[:, :num_outputs, :]
        Jz2 = Js[:, num_outputs:, :]

        m = 0.1  # margin
        mask = torch.logical_and(
            (1 - y).bool(),
            m - torch.norm(f, dim=1) < 0
        )
        if hessian_structure == "diag":
            Hs = torch.einsum("nij,nij->nj", Jz1, Jz1) + \
                 torch.einsum("nij,nij->nj", Jz2, Jz2) - \
                 2 * (
                         torch.einsum("nij,nij->nj", Jz1, Jz2) +
                         torch.einsum("nij,nij->nj", Jz2, Jz1)
                 )
            mask = mask.view(-1, 1).expand(**Hs.shape)
        elif hessian_structure == "full":
            Hs = torch.einsum("nij,nkl->njl", Jz1, Jz1) + \
                 torch.einsum("nij,nkl->njl", Jz2, Jz2) - \
                 2 * (
                         torch.einsum("nij,nkl->njl", Jz1, Jz2) +
                         torch.einsum("nij,nkl->njl", Jz2, Jz1)
                 )
            mask = mask.view(-1, 1, 1).expand(**Hs.shape)
        else:
            raise NotImplementedError

        Hs = Hs.masked_fill_(mask, 0.)
    else:
        raise NotImplementedError

    if agg == "sum":
        Hs = Hs.sum(dim=0)
    return Hs
