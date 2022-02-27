import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pl_bolts.datamodules import CIFAR10DataModule

from src import models as m
from src.models.train_helper import test_model


def train_model():
    model_name = "conv_net"

    model = m.ConvNet(lr=0.01)
    data = CIFAR10DataModule("./data", batch_size=32, num_workers=4, normalize=True)
    data.setup()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=1,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(save_dir="logs/", name=model_name),
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="min",
            ),
            callbacks.ModelCheckpoint(
                dirpath=f"models/{model_name}/",
                verbose=True,
                monitor="val_loss",
                mode="min",
            ),
        ],
    )

    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    trainer.test(dataloaders=data.test_dataloader(), ckpt_path="best")

    accuracies_train = test_model(data.dataset_train, data.dataset_train, model, "cpu")
    accuracies_val = test_model(data.dataset_train, data.dataset_val, model, "cpu")
    accuracies_test = test_model(data.dataset_train, data.dataset_test, model, "cpu")

    print(f"Train accuracy: {accuracies_train['precision_at_1']}")
    print(f"Val accuracy: {accuracies_val['precision_at_1']}")
    print(f"Test accuracy: {accuracies_test['precision_at_1']}")


if __name__ == "__main__":
    train_model()
