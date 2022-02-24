import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from src import data as d
from src import models as m
from src.data.get_dataset import get_dataset


def train_model():
    model_name = "conv_net"

    model = m.ConvNet(lr=0.01)
    data = get_dataset()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=3,
        log_every_n_steps=10,
        logger=TensorBoardLogger(save_dir="logs/", name=model_name),
        # callbacks=[
        #     callbacks.EarlyStopping(
        #         monitor="val_loss",
        #         min_delta=0.00,
        #         patience=5,
        #         verbose=False,
        #         mode="min",
        #     ),
        #     callbacks.ModelCheckpoint(
        #         dirpath=f"models/{model_name}/",
        #         verbose=True,
        #         monitor="val_loss",
        #         mode="min",
        #     ),
        # ],
    )

    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        # val_dataloaders=data.val_dataloader(),
    )

    # trainer.test(dataloaders=data.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    train_model()
