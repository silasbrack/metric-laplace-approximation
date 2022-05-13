from torch import nn


class ConvNet(nn.Module):
    def __init__(self, latent_dim=128, n_channels=3):
        super().__init__()
        if n_channels == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
            )
            self.linear = nn.Sequential(
                nn.Linear(4608, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
        elif n_channels == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
                nn.Flatten(),
            )
            self.linear = nn.Sequential(
                nn.Linear(6272, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
