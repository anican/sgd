import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=32*32*3, hidden_size=1024, output_size=10,
                 num_hidden_layers=2, batch_norm=True):
        super(MLP, self).__init__()

        self.num_hidden_layers = num_hidden_layers
        if num_hidden_layers == 0:
            self.classifier = nn.Linear(input_size, output_size)
        else:
            self.fc = [nn.Linear(input_size, hidden_size)]
            if batch_norm:
                self.fc.append(nn.BatchNorm1d(hidden_size))
            self.fc.append(nn.ReLU())
            if num_hidden_layers < 1:
                raise ValueError("Insufficient number of hidden layers!")
            for i in range(0, num_hidden_layers):
                self.fc.append(nn.Linear(hidden_size, hidden_size))
                if batch_norm:
                    self.fc.append(nn.BatchNorm1d(hidden_size))
                self.fc.append(nn.ReLU())
            self.layers = nn.Sequential(*self.fc)
            self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        if self.num_hidden_layers == 0:
            return self.classifier(x)
        outputs = self.layers(x)
        outputs = self.classifier(outputs)
        return outputs


def _test():
    model: nn.Module = MLP(input_size=784, hidden_size=256, num_hidden_layers=1,
                           output_size=10)
    inputs = torch.randn(1000, 784)
    print("inputs size", inputs.size())
    outputs = model(inputs)
    print("outputs size", outputs.size())
    targets = torch.randint(low=0, high=10, size=(1000,))
    loss = F.cross_entropy(outputs, targets)
    print("loss value", loss)
    params = model.state_dict().keys()
    print(params, '\n')


if __name__ == '__main__':
    _test()

