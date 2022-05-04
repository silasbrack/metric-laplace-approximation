from torch import nn


class ConvNet(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.Conv2d(3, 36, 5, 1),
            nn.ReLU(),
            nn.Conv2d(36, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(13824, latent_dim),
        )

    def forward(self, x):
        return self.model(x)
