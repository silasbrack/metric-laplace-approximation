from torch import nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, 5, 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,256, 5, 1),
            nn.ReLU(),
            # nn.AvgPool2d(kernel_size=2,stride=2),
            # nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(256,64),
        )

    def forward(self, x):
        return self.model(x)