from tqdm import tqdm
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from torch.optim import Adam


def train_metric(net, loader, device="cpu", epochs=50, lr=3e-4):
    miner = miners.MultiSimilarityMiner()
    contrastive_loss = losses.ContrastiveLoss()
    optim = Adam(net.parameters(), lr=lr)
    for epoch in tqdm(range(epochs)):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            output = net(x)
            hard_pairs = miner(output, y)
            loss = contrastive_loss(output, y, hard_pairs)
            loss.backward()
            optim.step()
    return net
