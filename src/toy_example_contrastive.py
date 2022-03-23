import torch
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_metric_learning import miners
from torch import nn
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from src.models.manual_hessian import ContrastiveHessianCalculator
from src.models.metric_diag_laplace import MetricDiagLaplace
import pickle

batch_size = 8
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

# la.fit(data.train_dataloader())

with open('models/laplace_metric.pkl', 'rb') as file:
    la = pickle.load(file)

print('Optimize')
# la.optimize_prior_precision(method='marglik', val_loader=data.val_dataloader())

# with open('models/laplace_metric.pkl', 'wb') as file:
#     pickle.dump(la, file)
    
predictions = la(next(iter(data.test_dataloader()))[0], pred_type='glm', link_approx='probit')
print(predictions)
print(la)

def plot_latent_space_ood(z_mu, z_sigma):
    # path, z_mu, z_sigma, labels, ood_z_mu, ood_z_sigma, ood_labels
# ):

    import numpy as np 
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    # max_ = np.max([np.max(z_sigma), np.max(ood_z_sigma)])
    # min_ = np.min([np.min(z_sigma), np.min(ood_z_sigma)])
    max_ = np.max(z_sigma)
    min_ = np.min(z_sigma)
    
    # normalize sigma
    z_sigma = ((z_sigma - min_) / (max_ - min_)) * 1
    # ood_z_sigma = ((ood_z_sigma - min_) / (max_ - min_)) * 1

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    for i, (z_mu_i, z_sigma_i) in enumerate(zip(z_mu, z_sigma)):

        ax.scatter(z_mu_i[0], z_mu_i[1], color="b")
        ellipse = Ellipse(
            (z_mu_i[0], z_mu_i[1]),
            width=z_sigma_i[0],
            height=z_sigma_i[1],
            fill=False,
            edgecolor="blue",
        )
        ax.add_patch(ellipse)

        if i > 500:
            ax.scatter(z_mu_i[0], z_mu_i[1], color="b", label="ID")
            break

    # for i, (z_mu_i, z_sigma_i) in enumerate(zip(ood_z_mu, ood_z_sigma)):

    #     ax.scatter(z_mu_i[0], z_mu_i[1], color="r")
    #     ellipse = Ellipse(
    #         (z_mu_i[0], z_mu_i[1]),
    #         width=z_sigma_i[0],
    #         height=z_sigma_i[1],
    #         fill=False,
    #         edgecolor="red",
    #     )
    #     ax.add_patch(ellipse)

    #     if i > 500:
    #         ax.scatter(z_mu_i[0], z_mu_i[1], color="r", label="OOD")
    #         break