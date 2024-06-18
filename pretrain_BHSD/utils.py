import numpy as np
import os
import pandas as pd
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

def info_message(message, *args, end="\n"):
    print(message.format(*args), end=end)

def create_result_df(args):
    train_result = pd.DataFrame({
            'Epoch': range(1, args.epochs+1),
            'Loss': None,
            'Dice coefficient': None,
            'MSE': None,
        })
    valid_result = pd.DataFrame({
            'Epoch': range(1, args.epochs+1),
            'Loss': None,
            'Dice coefficient': None,
            'MSE': None,
        })
    return train_result, valid_result

def write_values(epoch, result, dic):
    result.loc[epoch-1, 'Loss'] = dic['loss']
    result.loc[epoch-1, 'Dice coefficient'] = dic['dice']
    result.loc[epoch-1, 'MSE'] = dic['mse']
