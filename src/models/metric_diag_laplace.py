import torch
from laplace import DiagLaplace
from laplace.curvature import BackPackGGN
from pytorch_metric_learning import losses
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from src.models.manual_hessian import HessianCalculator, ContrastiveHessianCalculator


class MetricDiagLaplace(DiagLaplace):
    def __init__(self, model, likelihood, sigma_noise=1., prior_precision=1.,
                 prior_mean=0., temperature=1., backend=BackPackGGN, backend_kwargs=None):
        super().__init__(model, likelihood, sigma_noise, prior_precision,
                         prior_mean, temperature, backend, backend_kwargs)
        self.loss_fn = losses.ContrastiveLoss()
        self.hessian_calculator: HessianCalculator = ContrastiveHessianCalculator()

    def _curv_closure(self, X, y, N):
        a1, p, a2, n = lmu.get_all_pairs_indices(y, y)
        output = self.model(X)
        loss = self.loss_fn(output, y, (a1, p, a2, n))

        x1 = X[torch.cat((a1, a2))]
        x2 = X[torch.cat((p, n))]
        t = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))

        return loss, self.hessian_calculator.calculate_hessian(x1, x2, t, model=self.model, num_outputs=output.shape[-1])
