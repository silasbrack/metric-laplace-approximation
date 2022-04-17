import torch
from backpack import extend, backpack
from backpack.extensions import DiagGGNExact
from torch import nn


class HessianCalculator:
    def __init__(self,
                 model: nn.Sequential,
                 loss: nn.Module,
                 layers_to_estimate: nn.Module = None):
        self.model = model
        self.loss = loss
        if layers_to_estimate is None:
            layers_to_estimate = model
        extend(layers_to_estimate)
        extend(loss)
        self.layers_to_estimate = layers_to_estimate

    def compute_loss(self, *args):
        x, y = args
        return self.loss(self.model(x), y)

    def compute_batch(self, *args):
        loss = self.compute_loss(*args)
        with backpack(DiagGGNExact()):
            loss.backward()
        return torch.cat([p.diag_ggn_exact.data.flatten()
                          for p in self.layers_to_estimate.parameters()])

    def compute(self, loader):
        hessian = []
        for batch in loader:
            hessian.append(self.compute_batch(*batch))
        hessian = torch.mean(torch.stack(hessian, dim=0), dim=0)
        return hessian * len(loader.dataset)
