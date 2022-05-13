import logging
import pickle
from tkinter import N
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Subset, DataLoader
from pl_bolts.datamodules import MNISTDataModule

from src.data.cifar import CIFARData
from src.data import CIFAR100DataModule, SVHNDataModule
from src.models.conv_net import ConvNet


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    nn_samples = 16
    ood = "noise"

    batch_size = 16
    data = CIFARData("data/", batch_size, 4)
    data.setup()

    latent_dim = 15
    net = ConvNet(latent_dim, n_channels=1)
    # net_inference = net.linear
    net.to(device)

    net.load_state_dict(
        torch.load("models/laplace_model.ckpt", map_location=device)
    )
    h = torch.load("models/laplace_hessian.ckpt", map_location=device)

    mu_q = parameters_to_vector(net.linear.parameters())
    sigma_q = 1 / (h + 1e-6)

    def sample(parameters, posterior_scale, n_samples=16):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device=parameters.device)
        samples = samples * posterior_scale.reshape(1, n_params)
        return parameters.reshape(1, n_params) + samples

    logging.info("Sampling.")
    samples = sample(mu_q, sigma_q, nn_samples)

    device = "cpu"
    net = net.to(device)
    samples = samples.to(device)

    logging.info("Generating predictions from samples.")
    preds = generate_predictions_from_samples(
        data.test_dataloader(), samples, net, net.linear, device
    )
    preds = preds.detach().cpu()

    with open("preds_trained.pkl", "wb") as f:
        pickle.dump(
            {"means": preds.mean(dim=0), "vars": preds.var(dim=0)},
            f,
        )

    logging.info("Generating ood predictions from samples.")

    if ood == "cifar100":
        ood_data = CIFAR100DataModule("data/", batch_size, 4)
        ood_data.setup()
        test_set = ood_data.dataset_test
        subset_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        mask = torch.tensor(
            [test_set[i][1] in subset_classes for i in range(len(test_set))]
        )
        indices = torch.arange(len(test_set))[mask]
        ood_dataset = Subset(test_set, indices)
        ood_dataloader = DataLoader(
            ood_dataset, batch_size=batch_size, num_workers=4
        )
        preds_ood = generate_predictions_from_samples(
            ood_dataloader, samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "mnist":
        ood_data = MNISTDataModule("data/", batch_size, 4)
        ood_data.setup()
        preds_ood = generate_predictions_from_samples(
            ood_data.test_dataloader(), samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "svhn":
        ood_data = SVHNDataModule("data/", batch_size, 4)
        ood_data.prepare_data()
        ood_data.setup()
        preds_ood = generate_predictions_from_samples(
            ood_data.test_dataloader(), samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    elif ood == "noise":
        preds_ood = generate_fake_predictions_from_samples(
            data.test_dataloader(), samples, net, net.linear, device
        )
        preds_ood = preds_ood.detach().cpu()
    else:
        raise NotImplementedError

    with open("preds_ood_trained.pkl", "wb") as f:
        pickle.dump(
            {"means": preds_ood.mean(dim=0), "vars": preds_ood.var(dim=0)},
            f,
        )


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


if __name__ == "__main__":
    run()
