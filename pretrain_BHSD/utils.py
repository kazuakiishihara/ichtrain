import numpy as np
import os
import random
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def save_args(args, log_dir, file):
    log_file = os.path.join(log_dir, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(f'Script name: {file}\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

def info_message(message, *args, end="\n"):
    print(message.format(*args), end=end)
