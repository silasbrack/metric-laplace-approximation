import logging

######################################################################################################################
### This script is modified from the guide on pytorch distributed training https://github.com/seba-1511/dist_tuto.pth/
### https://pytorch.org/tutorials/intermediate/dist_tuto.html
######################################################################################################################
import os
from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule

from pytorch_metric_learning import losses, miners, testers
from pytorch_metric_learning.utils import distributed as pml_dist
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

logging.getLogger().setLevel(logging.INFO)


class Net(nn.Module):
    """Network architecture."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print(x.shape)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)
        x = x.view(-1, 500)
        # print(x.shape)
        return self.fc1(x)


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model, data_device):
    # dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    tester = testers.BaseTester(dataloader_num_workers=0, data_device=data_device)
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator, data_device):
    train_embeddings, train_labels = get_all_embeddings(train_set, model, data_device)
    test_embeddings, test_labels = get_all_embeddings(test_set, model, data_device)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print(
        "Validation set accuracy (Precision@1) = {}".format(
            accuracies["precision_at_1"]
        )
    )


def test_model(rank, train_set, test_set, model, epoch, data_device):
    if rank == 0:
        print("Computing validation set accuracy for epoch {}".format(epoch))
        accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
        test(train_set, test_set, model, accuracy_calculator, data_device)
    dist.barrier()


def run(rank, size, train_loader, val_loader):
    """Distributed Synchronous SGD Example"""
    print("Rank {} entering the 'run' function".format(rank))
    torch.manual_seed(1234)
    dist.barrier()
    ### use this if you have multiple GPUs ###
    # device = torch.device("cuda:{}".format(rank))
    batch_size = 32
    device = torch.device("cpu")
    model = Net()
    ### if you have multiple GPUs, set this to DDP(model.to(device), device_ids=[rank])
    model = DDP(model.to(device))
    # test_model(rank, train_dataset, val_dataset, model, "untrained", device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #####################################
    ### pytorch-metric-learning stuff ###
    loss_fn = losses.TripletMarginLoss()
    loss_fn = pml_dist.DistributedLossWrapper(loss=loss_fn, efficient=True)
    miner = miners.MultiSimilarityMiner()
    miner = pml_dist.DistributedMinerWrapper(miner=miner, efficient=True)
    ### pytorch-metric-learning stuff ###
    #####################################

    # num_batches = ceil(len(train_set.dataset) / float(bsz))
    num_batches = len(train_loader.dataset) // batch_size
    for epoch in range(1):
        epoch_loss = 0.0
        print("Rank {} starting epoch {}".format(rank, epoch))
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print(output.shape, target.shape)
            hard_pairs = miner(output, target)
            loss = loss_fn(output, target, hard_pairs)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(
                    "Rank {}, iteration {}, loss {}, num pos pairs {}, num neg pairs {}".format(
                        rank,
                        i,
                        loss.item(),
                        miner.miner.num_pos_pairs,
                        miner.miner.num_neg_pairs,
                    )
                )
            dist.barrier()

        print(
            "Rank {}, epoch {}, average loss {}".format(
                rank, epoch, epoch_loss / num_batches
            )
        )
        test_model(rank, train_loader, val_loader, model, epoch, device)


#######################################
### Set backend='nccl' if using GPU ###
#######################################
def init_processes(rank, size, fn, train_dataset, val_dataset, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, train_dataset, val_dataset)


if __name__ == "__main__":
    batch_size = 32
    # data = MNISTDataModule("./data", batch_size=batch_size)
    data = CIFAR10DataModule("./data", batch_size=batch_size)
    data.setup()

    size = 4
    processes = []
    for rank in range(size):
        p = Process(
            target=init_processes, args=(rank, size, run, data.train_dataloader(), data.val_dataloader())
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
