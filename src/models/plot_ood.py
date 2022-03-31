import logging
import pickle

import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning import losses, miners
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.models.manual_hessian import ContrastiveHessianCalculator
from src.models.train_metric import run as train_metric


def _curv_closure(model, miner, loss_fn, calculator, X, y):
    embeddings = model(X)

    a1, p, a2, n = miner(embeddings, y)
    loss = loss_fn(embeddings, y, (a1, p, a2, n))

    x1 = X[torch.cat((a1, a2))]
    x2 = X[torch.cat((p, n))]
    t = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))

    H = calculator.calculate_hessian(x1, x2, t, model=model, num_outputs=embeddings.shape[-1])

    return loss, H


def run():
    batch_size = 32

    data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=0, normalize=True)
    data.setup()
    loader = data.val_dataloader()

    logging.info("Finding MAP solution.")
    model = train_metric().model
    model.eval()

    loss_fn = losses.ContrastiveLoss()
    calculator = ContrastiveHessianCalculator()
    miner = miners.BatchEasyHardMiner(pos_strategy='easy',
                                      neg_strategy='easy',
                                      allowed_pos_range=(0.2, 1),
                                      allowed_neg_range=(0.2, 1))
    logging.info("Calculating Hessian.")
    hs = []
    for x, y in iter(loader):
        loss, h = _curv_closure(model, miner, loss_fn, calculator, x, y)
        hs.append(h)
    hs = torch.stack(hs, dim=0)
    h = torch.sum(hs, dim=0)

    mu_q = parameters_to_vector(model.parameters())
    sigma_q = 1 / (h + 1e-6)

    def sample(parameters, posterior_scale, n_samples=100):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device="cpu")
        samples = samples * posterior_scale.reshape(1, n_params)
        return parameters.reshape(1, n_params) + samples

    logging.info("Sampling.")
    samples = sample(mu_q, sigma_q, n_samples=16)

    logging.info("Generating predictions from samples.")
    preds = []
    for net_sample in samples:
        vector_to_parameters(net_sample, model.parameters())
        batch_preds = []
        for x, _ in loader:
            pred = model(x)
            batch_preds.append(pred)
        preds.append(torch.cat(batch_preds, dim=0))
    logging.info("Stacking.")
    preds = torch.stack(preds)

    logging.info("Saving.")
    with open("preds.pkl", "wb") as f:
        pickle.dump({"means": preds.mean(dim=0), "vars": preds.var(dim=0)}, f)

    preds_ood = []
    for net_sample in samples:
        vector_to_parameters(net_sample, model.parameters())
        batch_preds = []
        for x, _ in loader:
            x = x + torch.randn(x.shape)  # batch_size, n_channels, width, height
            pred = model(x)
            batch_preds.append(pred)
        preds_ood.append(torch.cat(batch_preds, dim=0))
    preds_ood = torch.stack(preds_ood)

    with open("preds_ood.pkl", "wb") as f:
        pickle.dump({"means": preds_ood.mean(dim=0), "vars": preds_ood.var(dim=0)}, f)


if __name__ == "__main__":
    run()
