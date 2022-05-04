import pickle

from matplotlib import pyplot as plt

from src.visualization.helper import calibration_curves


def run():
    with open("preds.pkl", "rb") as f:
        preds = pickle.load(f)
    with open("targets.pkl", "rb") as f:
        targets = pickle.load(f)
    with open("preds_ood.pkl", "rb") as f:
        preds_ood = pickle.load(f)
    with open("targets_ood.pkl", "rb") as f:
        targets_ood = pickle.load(f)

    ece, acc, conf, bin_sizes = calibration_curves(
        targets.detach().numpy(), preds["means"].detach().numpy()
    )

    fig, ax = plt.subplots()
    ax.plot(conf, acc)
    ax.set(
        xlim=[0, 1],
        ylim=[0, 1],
    )
    fig.savefig("calibration_curve_posthoc.png")


if __name__ == "__main__":
    run()
