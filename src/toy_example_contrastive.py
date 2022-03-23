from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning import miners
from torch import nn

from src.models.manual_hessian import ContrastiveHessianCalculator

batch_size = 128
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

miner = miners.MultiSimilarityMiner()

data = CIFAR10DataModule("./data", batch_size=batch_size, num_workers=0, normalize=True)
data.setup()

train_loader_iter = iter(data.train_dataloader())
x1, y1 = next(train_loader_iter)
x2, y2 = next(train_loader_iter)
y = (y1 == y2).int()

Hs = ContrastiveHessianCalculator().calculate_hessian(x1, x2, y, model=model, num_outputs=latent_size)
print(Hs.shape)
