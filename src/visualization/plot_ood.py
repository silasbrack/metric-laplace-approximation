import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# matplotlib.use('Qt5Agg')
import pickle
import numpy as np
import seaborn as sns

# ext = "_posthoc_cifar100"

with open(f"preds.pkl", "rb") as f:
    preds = pickle.load(f)
with open(f"preds_ood.pkl", "rb") as f:
    preds_ood = pickle.load(f)

means_preds = preds["means"].detach().numpy()
vars_preds = preds["vars"].detach().numpy()

means_preds_ood = preds_ood["means"].detach().numpy()
vars_preds_ood = preds_ood["vars"].detach().numpy()
limit = 100

fig, axs = plt.subplots(2,1, figsize=(4, 6))

## Plot scatter with uncertainty

axs[0].scatter(means_preds[:limit,0],means_preds[:limit,1], s=0.5, c="b", label="i.d.")
axs[0].scatter(means_preds_ood[:limit,0],means_preds_ood[:limit,1], s=0.5, c="r", label="o.o.d")

for i in range(limit):
    elp = Ellipse((means_preds[i,0],means_preds[i,1]), vars_preds[i,0], vars_preds[i,1], fc='None', edgecolor='b', lw=0.5)
    axs[0].add_patch(elp)
    elp = Ellipse((means_preds_ood[i,0],means_preds_ood[i,1]), vars_preds_ood[i,0], vars_preds_ood[i,1], fc='None', edgecolor='r', lw=0.5)
    axs[0].add_patch(elp)

# handles, labels = axs[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center')

axs[0].set(
    xlabel="Latent dim 1",
    ylabel="Latent dim 2",
)

## Plot variance density
id_density = vars_preds.flatten()
sns.kdeplot(id_density, ax=axs[1], color="b", label="i.d.")

ood_density = vars_preds_ood.flatten()
sns.kdeplot(ood_density, ax=axs[1], color="r", label="o.o.d")

axs[1].set(
    xlabel="Variance",
)
axs[1].legend()

fig.tight_layout()
fig.savefig("ood_plot.png")