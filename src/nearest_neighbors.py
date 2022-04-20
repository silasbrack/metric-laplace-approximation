import faiss
import torch
from torch.utils.data import TensorDataset, DataLoader
import faiss.contrib.torch_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_nearest_latent_neighbors(dataset, model, latent_size, num_neighbors):
    dataloader = DataLoader(dataset, batch_size=1024)

    y_hat = torch.cat([model(x.to(device)) for x, y in dataloader])
    
    if device == 'cpu':
        index = faiss.IndexFlatL2(latent_size)
    else:
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0

        flat_config = cfg
        resources = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(resources, latent_size, flat_config)
    
    index.add(y_hat)
    
    _, indices = index.search(y_hat, num_neighbors)

    x1s = []
    x2s = []
    ys = []
    for i in range(indices.shape[0]):
        x1, y1 = dataset[i]
        x1 = x1.expand((indices.shape[1], *x1.shape))
        y1 = torch.tensor(y1)
        y1 = y1.expand((indices.shape[1], *y1.shape))
        x2 = [dataset[indices[i, j]][0] for j in range(indices.shape[1])]
        y2 = [dataset[indices[i, j]][1] for j in range(indices.shape[1])]
        x2 = torch.stack(x2)
        y2 = torch.tensor(y2)
        x1s.append(x1)
        x2s.append(x2)
        ys.append((y1 == y2).int())
    x1s = torch.cat(x1s, dim=0)
    x2s = torch.cat(x2s, dim=0)
    ys = torch.cat(ys, dim=0)

    pair_dataset = TensorDataset(x1s, x2s, ys)
    return pair_dataset
