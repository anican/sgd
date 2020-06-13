from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    writer = SummaryWriter(log_dir=os.getcwd()+'/logging/test')

    for n_iter in tqdm(range(1000)):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    writer.close()
