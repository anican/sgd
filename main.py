#!/usr/bin/env python
import argparse
import os
from project import Project
from project import restore_model
import torch
from torch import nn
from utils import query_model
from utils import train_model
from utils import test_model


def pretrain():
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='alexnet')
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--batch_size_test', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--query_batch_size', type=int, default=1024)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    args = parser.parse_args()

    # experiment directories
    project = Project(args)
    checkpoint_dir = project.get_dir('checkpoints')

    # hyperparameters and experiment variables
    arch = args.arch
    epochs = args.epochs
    print_interval = args.print_interval
    seed = args.seed

    # dataLoaders and devices
    project.prepare_data()
    train_dataloader = project.train_dataloader()

    val_dataloader = project.val_dataloader()
    test_dataloader = project.test_dataloader()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model and seed
    torch.manual_seed(seed=seed)
    model = project.configure_model()
    if torch.cuda.device_count() > 1:
        print("using", torch.cuda.device_count(), "gpus!")
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer, scheduler, logger
    logger = project.configure_logger()
    optimizer, scheduler = project.configure_optimizers(model=model)

    # training and validation
    train_model(model=model, optimizer=optimizer, device=device,
                print_interval=print_interval,
                epochs=epochs, train_dataloader=train_dataloader,
                val_dataloader=val_dataloader, logger=logger)
    project.save_model(model, optimizer)
    logger.close()


def query():
    model, optimizer, args = restore_model(os.getcwd() + '/checkpoints/alexnet/alexnet_cifar10_ep_1.pt')
    seed = args.seed
    project = Project(args)
    project.prepare_data()
    query_dataloader = project.query_dataloader()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed=seed)
    if torch.cuda.device_count() > 1:
        print("using", torch.cuda.device_count(), "gpus!")
        model = nn.DataParallel(model)
    model.to(device)
    norms, alphas = query_model(model=model, query_dataloader=query_dataloader,
                                device=device, optimizer=optimizer, norm_sample_count=1)
    print(len(norms), len(alphas))
    print(norms[0].shape, alphas[0].shape)


if __name__ == '__main__':
    # pretrain()
    query()

