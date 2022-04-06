from torch import nn


class LinearNet(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x):
        return self.model(x)
