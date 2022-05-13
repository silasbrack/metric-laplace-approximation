from torch import nn
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from src.data.cifar import CIFARData
from src.models.conv_net import ConvNet
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns


latent_dim = 2
model = ConvNet(latent_dim=2, n_channels=1)
inference_model = model.linear

net_state_dict = torch.load(
    "models/laplace_model.ckpt", map_location=torch.device("cpu")
)
model.load_state_dict(net_state_dict)
H = torch.load("models/laplace_hessian.ckpt", map_location=torch.device("cpu"))

batch_size = 8
data = CIFARData("data/", batch_size, 4)
data.setup()
loader = data.val_dataloader()

mu_q = parameters_to_vector(inference_model.parameters())
sigma_q = 1 / (H + 1e-6)


def generate_predictions_from_samples(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for x, _ in iter(loader):
            x = x.to(device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


def generate_fake_predictions_from_samples(
    loader, weight_samples, full_model, inference_model=None, device="cpu"
):
    if inference_model is None:
        inference_model = full_model

    preds = []
    for net_sample in weight_samples:
        vector_to_parameters(net_sample, inference_model.parameters())
        sample_preds = []
        for x, _ in iter(loader):
            x = torch.randn(x.shape, device=device)
            pred = full_model(x)
            sample_preds.append(pred)
        preds.append(torch.cat(sample_preds, dim=0))
    return torch.stack(preds, dim=0)


def sample(parameters, posterior_scale, n_samples=16):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device="cpu")
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


samples = sample(mu_q, sigma_q, n_samples=16)

preds = generate_predictions_from_samples(
    data.test_dataloader(), samples, model, inference_model
)
preds = preds.detach().cpu()

preds_ood = []
for net_sample in samples:
    vector_to_parameters(net_sample, inference_model.parameters())
    batch_preds = []
    for x, _ in loader:
        x = torch.randn(x.shape)  # batch_size, n_channels, width, height
        pred = model(x)
        batch_preds.append(pred)
    preds_ood.append(torch.cat(batch_preds, dim=0))
preds_ood = torch.stack(preds_ood)

means_preds = preds.mean(dim=0).detach().numpy()
vars_preds = preds.var(dim=0).detach().numpy()

means_preds_ood = preds_ood.mean(dim=0).detach().numpy()
vars_preds_ood = preds_ood.var(dim=0).detach().numpy()

limit = 100

fig, axs = plt.subplots(2, 1, figsize=(15, 15))

## Plot scatter with uncertainty

axs[0].scatter(
    means_preds[:limit, 0], means_preds[:limit, 1], s=0.5, c="b", label="i.d."
)
axs[0].scatter(
    means_preds_ood[:limit, 0],
    means_preds_ood[:limit, 1],
    s=0.5,
    c="r",
    label="o.o.d",
)

for i in range(limit):
    elp = Ellipse(
        (means_preds[i, 0], means_preds[i, 1]),
        vars_preds[i, 0],
        vars_preds[i, 1],
        fc="None",
        edgecolor="b",
        lw=0.5,
    )
    axs[0].add_patch(elp)
    elp = Ellipse(
        (means_preds_ood[i, 0], means_preds_ood[i, 1]),
        vars_preds_ood[i, 0],
        vars_preds_ood[i, 1],
        fc="None",
        edgecolor="r",
        lw=0.5,
    )
    axs[0].add_patch(elp)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center")

## Plot variance density
id_density = vars_preds.flatten()
sns.kdeplot(id_density, ax=axs[1], color="b")

ood_density = vars_preds_ood.flatten()
sns.kdeplot(ood_density, ax=axs[1], color="r")

fig.savefig("test.png")
