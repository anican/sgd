import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


if __name__ == '__main__':
    x = torch.randn(50, 1, 28, 28)
    model = MLP()
    outputs = model(x)
    print(outputs.shape)


