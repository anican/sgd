import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def write_tail_data(tail_data, tail_path: str):
    """
    Writes the given data to the given path in a format such that it can be reconstructed for future analysis.

    :param tail_data: 3-tuple of list of tensors (iterations, norms, alphas)
    :param tail_path: file path where the tail-index relevant data is stored
    :return:
    """
    iterations, grad_norms, alphas = zip(*tail_data)
    tail_data_history = {"Iterations": iterations, "SGD Norms":grad_norms, "Alpha Estimates":alphas}
    torch.save(tail_data_history, tail_path)


def estimate_alpha(Z):
    X = Z.reshape(-1)
    X = X[X.nonzero()]
    K = len(X)
    if len(X.shape) > 1:
        X = X.squeeze()
    K1 = int(np.floor(np.sqrt(K)))
    K2 = K1
    X = X[:K1*K2].reshape((K2, K1))
    Y = X.sum(1)
    # X = X.cpu().clone(); Y = Y.cpu().clone()
    a = torch.log(torch.abs(Y)).mean()
    b = (torch.log(torch.abs(X[:int(K2/4), :])).mean() +
         torch.log(torch.abs(X[int(K2/4):int(K2/2), :])).mean() +
         torch.log(torch.abs(X[int(K2/2):int(3*K2/4), :])).mean() +
         torch.log(torch.abs(X[int(3*K2/4):, :])).mean())/4
    return np.log(K1)/(a-b).item()


def get_sgd_noise(model: nn.Module, curr_device, optimizer, query_dataloader):
    """

    :param model:
    :param curr_device:
    :param optimizer:
    :param query_dataloader:
    :return:
    """
    gc.collect()
    # We do NOT want to be training on the full gradients, just calculating them!!!!
    model.eval()
    grads, sizes = [], []
    for batch_idx, (data, target) in enumerate(query_dataloader):
        data, target = data.to(curr_device), target.to(curr_device)
        optimizer.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        grad = [param.grad.cpu().clone() for param in model.parameters()]
        # grad = [p.grad.clone() for p in model.parameters()]
        size = data.shape[0]
        grads.append(grad)
        sizes.append(size)

    flat_grads = []
    for grad in grads:
        flat_grads.append(torch.cat([g.reshape(-1) for g in grad]))
    full_grads = torch.zeros(flat_grads[-1].shape)
    # Exact_Grad = torch.zeros(Flat_Grads[-1].shape).cuda()
    for g, s in zip(flat_grads, sizes):
        full_grads += g * s
    full_grads /= np.sum(sizes)
    gc.collect()
    flat_grads = torch.stack(flat_grads)
    sgd_noise = (flat_grads-full_grads).cpu()
    # Grad_noise = Flat_Grads-Exact_Grad
    return full_grads, sgd_noise
