import torch
from torch import nn
from models import MLP


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MLP()
    model = nn.DataParallel(model)
    print(model)
