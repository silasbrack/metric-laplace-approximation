import fire
from laplace import Laplace
from pl_bolts.datamodules import CIFAR10DataModule

from src.models.train_metric_learning import run as train_model


def run():
    model = train_model(epochs=1)

    data = CIFAR10DataModule("./data", batch_size=32, num_workers=4, normalize=True)
    data.setup()

    la = Laplace(model, 'classification',
                 subset_of_weights='all',
                 hessian_structure='diag')
    la.fit(data.train_dataloader())
    la.optimize_prior_precision(method='CV', val_loader=data.val_dataloader())
    print(la)


if __name__ == "__main__":
    fire.Fire(run())

