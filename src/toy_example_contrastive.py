import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning import miners
from torch import nn
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from src.models.manual_hessian import ContrastiveHessianCalculator
from src.models.metric_diag_laplace import MetricDiagLaplace

batch_size = 32
latent_size = 3

model = nn.Sequential(
    nn.Conv2d(3, 16, 3, 1),
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Flatten(),
    nn.Linear(6272, latent_size),
)
num_params = sum(p.numel() for p in model.parameters())

miner = miners.BatchEasyHardMiner(pos_strategy="all", neg_strategy="all")

data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=0, normalize=True)
data.setup()

# train_loader_iter = iter(data.train_dataloader())
# x1, y1 = next(train_loader_iter)
# x2, y2 = next(train_loader_iter)
# y = (y1 == y2).int()

# Hs = torch.Tensor()
# for x, labels in data.train_dataloader():
#     a1, p, a2, n = lmu.get_all_pairs_indices(labels, labels)
#     x1 = x[torch.cat((a1, a2))]
#     x2 = x[torch.cat((p, n))]
#     y = torch.cat((torch.ones(p.shape[0]), torch.zeros(n.shape[0])))
#     Hs = ContrastiveHessianCalculator().calculate_hessian(x1, x2, y, model=model, num_outputs=latent_size)
#     print(Hs.shape)

la = MetricDiagLaplace(
    model,
    "regression",
    # subset_of_weights="last_layer",
    # hessian_structure="diag"
)
la.fit(data.train_dataloader())
