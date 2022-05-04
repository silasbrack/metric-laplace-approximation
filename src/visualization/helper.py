import numpy as np


def calibration_curves(targets, probs, bins=10, fill_nans=False):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)[:, None]

    real_probs = np.zeros((bins,))
    pred_probs = np.zeros((bins,))
    bin_sizes = np.zeros((bins,))

    _, lims = np.histogram(confidences, range=(0.0, 1.0), bins=bins)
    for i in range(bins):
        lower, upper = lims[i], lims[i + 1]
        mask = (lower <= confidences) & (confidences < upper)

        targets_in_range = targets[mask]
        preds_in_range = preds[mask]
        probs_in_range = confidences[mask]
        n_in_range = preds_in_range.shape[0]

        range_acc = (
            np.sum(targets_in_range == preds_in_range) / n_in_range
            if n_in_range > 0
            else 0
        )
        range_prob = (
            np.sum(probs_in_range) / n_in_range if n_in_range > 0 else 0
        )

        real_probs[i] = range_acc
        pred_probs[i] = range_prob
        bin_sizes[i] = n_in_range

    bin_weights = bin_sizes / np.sum(bin_sizes)
    ece = np.sum(np.abs(real_probs - pred_probs) * bin_weights)

    if fill_nans:
        return ece, real_probs, pred_probs, bin_sizes
    return ece, real_probs[bin_sizes > 0], pred_probs[bin_sizes > 0], bin_sizes
