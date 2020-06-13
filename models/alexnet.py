import torch
import torch.nn.functional as F
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, input_height=32, input_width=32, input_channels=3, ch=64,
                 output_size=10):
        super(AlexNet, self).__init__()
        self.best_accuracy = 0

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.output_size = output_size

        self.features = nn.Sequential(
            nn.Conv2d(3, out_channels=ch, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, self.output_size)
        )

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = outputs.squeeze()
        outputs = self.classifier(outputs)
        return outputs  # {"logits": outputs, "probas": F.softmax(outputs, dim=1)}


def _test():
    model: nn.Module = AlexNet()
    # Imagine 50 sample images
    inputs = torch.randn(50, 3, 32, 32)
    print("inputs size", inputs.size())
    outputs = model(inputs)
    print("outputs size", outputs.size())
    targets = torch.randint(low=0, high=10, size=(50,))
    loss = F.cross_entropy(outputs, targets)
    print("loss value", loss)
    params = model.state_dict().keys()
    print(params, '\n')


if __name__ == '__main__':
    _test()
