import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src import data as d
from src import models as m
from src.data.get_dataset import get_dataset


def train_model():
    model_name = "conv_net"

    model = m.ConvNet(lr=0.01)
    data = get_dataset()

    trainer = pl.Trainer(
        gpus=0,
        max_epochs=1,
        log_every_n_steps=10,
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

    def get_all_embeddings(dataset, model, data_device):
        # dataloader_num_workers has to be 0 to avoid pid error
        # This only happens when within multiprocessing
        tester = testers.BaseTester(dataloader_num_workers=0, data_device=data_device)
        return tester.get_all_embeddings(dataset, model)

    def test(train_set, test_set, model, accuracy_calculator, data_device):
        train_embeddings, train_labels = get_all_embeddings(train_set.dataset, model, data_device)
        test_embeddings, test_labels = get_all_embeddings(test_set.dataset, model, data_device)
        train_labels = train_labels.squeeze(1)
        test_labels = test_labels.squeeze(1)
        accuracies = accuracy_calculator.get_accuracy(
            test_embeddings, train_embeddings, test_labels, train_labels, False
        )
        return accuracies

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    accuracies_train = test(data.train_dataloader(), data.train_dataloader(), model, accuracy_calculator, 'cpu')
    accuracies_val = test(data.train_dataloader(), data.val_dataloader(), model, accuracy_calculator, 'cpu')
    accuracies_test = test(data.train_dataloader(), data.test_dataloader(), model, accuracy_calculator, 'cpu')

    print(f"Train accuracy: {accuracies_train['precision_at_1']}")
    print(f"Val accuracy: {accuracies_val['precision_at_1']}")
    print(f"Test accuracy: {accuracies_test['precision_at_1']}")


if __name__ == "__main__":
    train_model()
