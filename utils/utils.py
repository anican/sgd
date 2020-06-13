import math
import time
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .sgd_utils import get_sgd_noise
from .sgd_utils import estimate_alpha


def train_model(model: nn.Module,
                optimizer: optim.SGD,
                epochs: int,
                device: torch.device,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                logger: SummaryWriter,
                print_interval: int = 50):
    """

    :param model:
    :param optimizer:
    :param epochs:
    :param device:
    :param train_dataloader:
    :param val_dataloader: TODO: make optional
    :param logger: TODO: make optional
    :param print_interval: TODO: add as argument to argparser
    :return:
    """
    print("Training Model...")
    start = time.time()
    train_step, val_step = 0, 0
    for epoch in tqdm(range(epochs)):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # train step + backward step
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            # sgd noise step
            # _, sgd_noise = get_sgd_noise(model, device, optimizer, query_dataloader)
            # print('noise shape', sgd_noise.shape)
            # noise_norm = torch.norm(sgd_noise, dim=1)
            # alpha_hat = estimate_alpha(sgd_noise)
            # print('grad_norm', noise_norm.shape)
            # print('alpha_hat', alpha_hat)
            # # start train step logging here
            # logger.add_scalar('train/loss', loss.item(), train_step)
            # # TODO: how to log noise norm tensor?
            # logger.add_scalar('train/alpha', alpha_hat)
            train_step += 1
            if not batch_idx % print_interval:
                print_train_step(epoch, epochs, batch_idx, len(train_dataloader),
                                 loss.item())
            # end of train step logging
        model.eval()
        with torch.no_grad():
            correct, samples = 0, 0
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data, target = data.to(device), target.to(device)
                # validation step
                logits = model(data)
                target_hat = torch.argmax(logits, dim=1)
                val_loss = F.cross_entropy(logits, target)
                correct += torch.sum(target == target_hat).item()
                samples += len(target)
                # start val step logging here
                logger.add_scalar('Loss/val', val_loss.item(), val_step)
                val_step += 1
                # end of val step logging
        # scheduler.step(epoch)
        # start val epoch end logging here
        val_acc = correct / (samples * 1.0)
        logger.add_scalar('Acc/val', val_acc, epoch)
        print_validation_step(curr_epoch=epoch, epochs=epochs, val_acc=val_acc)
        # end val epoch logging here
    end = time.time()
    print('Total Training Time: %.2f min\n' % ((end - start) / 60))


def query_model(model: nn.Module, query_dataloader: DataLoader, device: torch.device,
                optimizer: optim.SGD, norm_sample_count: int):
    """

    :param model:
    :param query_dataloader:
    :param device:
    :param optimizer:
    :param norm_sample_count: number of norm samples to be calculated.
    :return:
    """
    noise_norms, alphas = [], []
    print('len query', len(query_dataloader))
    query_steps = int(math.ceil(norm_sample_count / len(query_dataloader)))
    print('query steps', query_steps)
    for ii in range(query_steps):
        # SGD calculation step
        # TODO: calculating the alphas at the proper step?
        # TODO: logger the alphas
        _, sgd_noise = get_sgd_noise(model, device, optimizer, query_dataloader)
        noise_norm = torch.norm(sgd_noise, dim=1)
        alpha_hat = estimate_alpha(sgd_noise)
        noise_norms.append(noise_norm)
        alphas.append(alpha_hat)
    return noise_norms, alphas


def test_model(model: nn.Module, test_dataloader: DataLoader, device: torch.device):
    print("Testing Model..")
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        correct, samples = 0, 0
        for batch_idx, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            # test step
            logits = model(data)
            target_hat = torch.argmax(logits, dim=1)
            loss = F.cross_entropy(logits, target).item()
            test_loss += loss
            correct += torch.sum(target == target_hat).item()
            samples += len(target)
        test_loss /= len(test_dataloader)
        test_acc = correct / (samples * 1.0)
        return {'test_loss': test_loss, 'test_acc': test_acc}


def print_train_step(curr_epoch: int, epochs: int, batch_idx: int, dataloader_len: int,
                     loss_val: int):
    train_step_stats = (curr_epoch+1, epochs, batch_idx, dataloader_len, loss_val)
    print('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' % train_step_stats)


def print_validation_step(curr_epoch: int, epochs: int, val_acc: float):
    val_step_stats = (curr_epoch+1, epochs, val_acc)
    print('Epoch End: %03d/%03d | Accuracy: %.3f%%' % val_step_stats)

