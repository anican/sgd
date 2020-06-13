from argparse import Namespace
from dataclasses import dataclass
from models import AlexNet, MLP, VGG, ResNet18, ResNet34
from pathlib import Path
import torch
from torch import nn
from torch import optim
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms


@dataclass
class Project(object):
    """This class is used to keep track of all the relevant directories for
    cleanly managing a deep learning exeeriment.

    Attributes:
        BASE_PATH: Path, base directory for all project activities.
        CHECKPOINS_PATH: Path, directory for storing model checkpoints
        DATASET_PATH: Path, directory for storing all datasets related to
            project experiments.
        IMAGES_PATH: Path, directory for storing images from the experiments.
            This is the place to store any samples or pictures not related to
            logging info pertaining to train/val/test metrics.
        LOG_PATH: Path, directory for storing all logger information (Comet.ml,
            tensorboard, neptune, etc.)

    """
    BASE_PATH: Path = Path(__file__).parents[0]
    CHECKPOINTS_PATH: Path = BASE_PATH / 'checkpoints'
    DATASET_PATH: Path = BASE_PATH / 'dataset'
    LOG_PATH: Path = BASE_PATH / 'logging'
    IMAGES_PATH: Path = BASE_PATH / 'images'

    def __init__(self, hparams: Namespace):
        self.hparams = hparams
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.CHECKPOINTS_PATH.mkdir(exist_ok=True)
        self.DATASET_PATH.mkdir(exist_ok=True)
        self.LOG_PATH.mkdir(exist_ok=True)
        self.IMAGES_PATH.mkdir(exist_ok=True)

    def get_dir(self, dir_type: str = None) -> str:
        if dir_type == 'checkpoints':
            return str(self.CHECKPOINTS_PATH)
        elif dir_type == 'dataset':
            return str(self.DATASET_PATH)
        elif dir_type == 'images':
            return str(self.IMAGES_PATH)
        elif dir_type == 'log':
            return str(self.LOG_PATH)
        else:
            return str(self.BASE_PATH)

    def prepare_data(self):
        dataset: str = self.hparams.dataset
        dataset_dir = self.get_dir('dataset')
        if dataset == 'mnist':
            self.train_data = MNIST(dataset_dir, train=True, download=True,
                                    transform=transforms.ToTensor())
            eval_data = MNIST(dataset_dir, train=False, download=True,
                              transform=transforms.ToTensor())
            self.val_data, self.test_data = random_split(eval_data, [5000, 5000])
        elif dataset == 'cifar10':
            self.train_data = CIFAR10(dataset_dir, train=True, download=True,
                                      transform=None)
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            self.train_data = CIFAR10(dataset_dir, train=True, download=True,
                                      transform=transform_train)
            eval_data = CIFAR10(dataset_dir, train=False, download=True,
                                transform=transform_test)
            self.val_data, self.test_data = random_split(eval_data, [5000, 5000])
        else:
            raise ValueError("Error: Invalid choice of dataset!")

    def query_dataloader(self):
        batch_size = self.hparams.query_batch_size
        num_workers = cpu_count()
        return DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)

    def train_dataloader(self):
        batch_size = self.hparams.batch_size
        num_workers = cpu_count()
        return DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)

    def val_dataloader(self):
        batch_size = self.hparams.batch_size_test
        num_workers = cpu_count()
        return DataLoader(self.val_data, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)

    def test_dataloader(self):
        batch_size = self.hparams.batch_size_test
        num_workers = cpu_count()
        return DataLoader(self.test_data, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers)

    def configure_logger(self):
        # TODO: configure the comment string using the hparams
        # comment = dataset + '_' + arch + '_lr_' + lr
        arch: str = self.hparams.arch
        comment: str = ""
        log_dir: str = str(self.LOG_PATH / arch)
        return SummaryWriter(comment=comment, log_dir=log_dir)

    def configure_model(self):
        """

        :return:
        """
        arch: str = self.hparams.arch
        batch_norm = self.hparams.batch_norm
        dataset: str = self.hparams.dataset
        hidden_layers: int = self.hparams.hidden_layers
        hidden_size: int = self.hparams.hidden_size
        if arch == 'mlp':
            if dataset == 'mnist':
                return MLP(input_size=784, hidden_size=hidden_size,
                           num_hidden_layers=hidden_layers, batch_norm=batch_norm)
            elif dataset == 'cifar10':
                return MLP(hidden_size=hidden_size, num_hidden_layers=hidden_layers,
                           batch_norm=batch_norm)
            else:
                raise ValueError('invalid dataset specification!')
        elif arch == 'alexnet':
            return AlexNet()
        elif arch == 'vgg11':
            return VGG(vgg_name='VGG11')
        elif arch == 'vgg13':
            return VGG(vgg_name='VGG13')
        elif arch == 'resnet18':
            return ResNet18()
        elif arch == 'resnet34':
            return ResNet34()
        else:
            raise ValueError('Unsupported model!')

    def configure_optimizers(self, model: nn.Module):
        gamma = self.hparams.gamma
        lr = self.hparams.lr
        momentum = self.hparams.momentum
        weight_decay = self.hparams.weight_decay
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
        return [optimizer, scheduler]

    def save_model(self, model: nn.Module, optimizer: torch.optim.SGD):
        arch = self.hparams.arch
        dataset = self.hparams.dataset
        epochs = self.hparams.epochs
        comment = "%s_%s_ep_%d.pt" % (arch, dataset, epochs)
        save_path = self.CHECKPOINTS_PATH / arch
        save_path.mkdir(exist_ok=True)
        save_path = save_path / comment
        print('testing save...', save_path)
        torch.save({'epoch': epochs, 'hparams': self.hparams,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, save_path)


def restore_model(file_path: str):
    info = torch.load(file_path)
    arch: str = info['hparams'].arch
    batch_norm = info['hparams'].batch_norm
    dataset: str = info['hparams'].dataset
    hidden_layers: int = info['hparams'].hidden_layers
    hidden_size: int = info['hparams'].hidden_size
    if arch == 'mlp' and dataset == 'mnist':
        model: nn.Module = MLP(input_size=784, hidden_size=hidden_size,
                               num_hidden_layers=hidden_layers, batch_norm=batch_norm)
    else:
        model: nn.Module = MLP(hidden_size=hidden_size,
                               num_hidden_layers=hidden_layers, batch_norm=batch_norm)

    model.load_state_dict(info['model_state_dict'])
    lr = info['hparams'].lr
    momentum = info['hparams'].momentum
    weight_decay = info['hparams'].weight_decay
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)
    optimizer.load_state_dict(info['optimizer_state_dict'])
    # model.eval()
    return model, optimizer, info['hparams']
