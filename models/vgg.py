import torch
from torch import nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
    Deep convolutional neural network from Oxford University. Achieved second
    place in the ImageNet competition 2014. Four varieties of VGG are provided.

    This particular model is adjusted for use on the CIFAR10 dataset.
    """
    def __init__(self, vgg_name, batch_norm=True, num_classes=10):
        super(VGG, self).__init__()
        self.features = VGG._make_layers(cfg[vgg_name], batch_norm)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, get_features=False):
        out = self.features(x)
        if get_features:
            out_flat = out.view(out.size(0), -1)
            out = self.classifier(out_flat)
            return out, out_flat
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    @staticmethod
    def _make_layers(param_spec: list, batch_norm: bool):
        layers = []
        in_channels = 3
        for x in param_spec:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1)]
                # nn.init.xavier_uniform_(layers[-1].weight.data) # self added
                # nn.init.xavier_uniform_(layers[-1].bias.data) # self added
                if batch_norm:
                    layers += [nn.BatchNorm2d(x)]
                layers += [nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def _test():
    net = VGG(vgg_name='VGG13')
    x = torch.randn(50, 3, 32, 32)
    y = net(x)
    print(y.size())
    print(net)


if __name__ == '__main__':
    _test()
