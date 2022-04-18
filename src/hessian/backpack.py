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

    # def compute(self, loader):
    #     hessian = []
    #     for batch in loader:
    #         hessian.append(self.compute_batch(*batch))
    #     hessian = torch.mean(torch.stack(hessian, dim=0), dim=0)
    #     return hessian * len(loader.dataset)
    def compute(self, loader):
        hessian = []
        for batch in loader:
            hessian.append(self.compute_batch(*batch) * batch[0].shape[0])
        hessian = torch.sum(torch.stack(hessian, dim=0), dim=0)
        return hessian


class ContrastiveHessianCalculator:
    def __init__(self,
                 model: nn.Sequential,
                 layers_to_estimate: nn.Module = None):
        self.model = model
        self.loss = nn.MSELoss()
        if layers_to_estimate is None:
            layers_to_estimate = model
        self.layers_to_estimate = layers_to_estimate
        extend(self.layers_to_estimate)
        extend(self.loss)

    def compute_batch(self, x1, x2, y):
        hessian1_rmse = self.compute_rmse(x1, y)
        hessian2_rmse = self.compute_rmse(x2, y)
        return


    def compute_rmse(self, *args):
        x, y = args
        loss = self.loss(self.model(x), y)
        with backpack(DiagGGNExact()):
            loss.backward()
        return torch.cat([p.diag_ggn_exact.data.flatten()
                          for p in self.layers_to_estimate.parameters()]) / 2

    def compute(self, loader):
        hessian = []
        for batch in loader:
            hessian.append(self.compute_batch(*batch) * batch[0].shape[0])
        hessian = torch.sum(torch.stack(hessian, dim=0), dim=0)
        return hessian
