import logging

import torch
from tqdm import tqdm
from pytorch_metric_learning import miners
from pytorch_metric_learning import losses
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim import Adam

from src.data.cifar import CIFARData
from src.hessian.rowwise import ContrastiveHessianCalculator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'{device=}')

def compute_kl_term(mu_q, sigma_q):
    """
    https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/
    """
    k = len(mu_q)
    return 0.5 * (
            - torch.log(sigma_q)
            - k
            + torch.dot(mu_q, mu_q)
            + torch.sum(sigma_q)
    )
    # k = len(mu_q)
    # return 0.5 * (
    #         torch.log(1.0 / (sigma_q + 1e-6) + 1e-6)
    #         - k
    #         + torch.matmul(mu_q.T, mu_q)
    #         + torch.sum(sigma_q)
    #     )


def sample_neural_network_wights(parameters, posterior_scale, n_samples=32):
    n_params = len(parameters)
    samples = torch.randn(n_samples, n_params, device=parameters.device)
    samples = samples * posterior_scale.reshape(1, n_params)
    return parameters.reshape(1, n_params) + samples


def run():
    contrastive_loss = losses.ContrastiveLoss()
    miner = miners.MultiSimilarityMiner()
    hessian_calculator = ContrastiveHessianCalculator()

    latent_dim = 2
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28*1, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, latent_dim),
    )
    net.to(device)
    
    # hessian_calculator.init_model(net)
    
    num_params = sum(p.numel() for p in net.parameters())

    optim = Adam(net.parameters(), lr=3e-4)

    batch_size = 16
    data = CIFARData("data/", batch_size, 4)
    data.setup()
    loader = data.val_dataloader()

    epochs = 4
    h = 1e10 * torch.ones((num_params,), device=device)
    F = 3
    kl_weight = 0.7
    
    for epoch in range(epochs):
        print(f"{epoch=}")
        epoch_losses = []
        train_laplace = epoch % F == 0 
        print(f'{train_laplace=}')
        
        for x, y in tqdm(loader):
            x, y = x.to(device), y.to(device)
            
            optim.zero_grad()
            
            # print('Training laplace')
            mu_q = parameters_to_vector(net.parameters())
            sigma_q = torch.abs(1 / (h + 1e-6))

            kl = compute_kl_term(mu_q, sigma_q)
            
            # print(f"{sigma_q=}")
            sampled_nn = sample_neural_network_wights(mu_q, sigma_q, n_samples=10)
            # print(f"{sampled_nn=}")
            
            con_losses = []
            if train_laplace:
                h = []
                
            for nn_i in sampled_nn:
                # print(f"{nn_i=}")
                vector_to_parameters(nn_i, net.parameters())

                output = net(x)
                hard_pairs = miner(output, y)
                # print(f"{hard_pairs=}")
                if train_laplace:
                    hessian_batch = hessian_calculator.compute_batch_pairs(net, output, x, y, hard_pairs)

                    # Adjust hessian to the batch size
                    hessian_batch = hessian_batch / batch_size * data.size
                    
                    h.append(hessian_batch)
                
                # print(f"{hessian_batch=}")
                con_loss = contrastive_loss(output, y, hard_pairs)
                con_losses.append(con_loss)
            
            # Add identity to h
            if train_laplace:
                h = torch.stack(h).mean(dim=0) if len(h) > 1 else h[0]
                h += 1
            
            con_loss = torch.stack(con_losses).mean(dim=0)
            # print(f"{h=}")
            # print(f"kl={kl.mean()}")
            
            # print(f'Loss pre KL = {con_loss}')
            
            loss = con_loss + kl.mean() * kl_weight
            # print(f'Loss post KL = {loss}')
            
            # Reassign parameters
            vector_to_parameters(mu_q, net.parameters())
                    
            # con_loss = torch.stack(con_loss).mean(dim=0)
            # print(f"Pre h={torch.stack(h)}")
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
            
        loss = torch.mean(torch.tensor(epoch_losses))
        logging.info(f"{loss=} for {epoch=}")

    torch.save(net.state_dict(), f='models/laplace_model.ckpt')
    torch.save(h, f='models/laplace_hessian.ckpt')
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
