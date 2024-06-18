import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
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

def metrics(label_li, prob_li):
    auc = roc_auc_score(label_li, prob_li)
    
    # Youden index
    fpr, tpr, thresholds = roc_curve(label_li, prob_li)
    youden_index = tpr - fpr
    optimal_threshold_index = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_threshold_index]

    pred_label = [1 if prob >= optimal_threshold else 0 for prob in prob_li]
    conf_matrix = confusion_matrix(label_li, pred_label)
    tn, fp, fn, tp = conf_matrix.ravel()

    acc = accuracy_score(label_li, pred_label)
    f1 = f1_score(label_li, pred_label)
    sen = recall_score(label_li, pred_label)
    spe = tn / (tn + fp)
    pre = precision_score(label_li, pred_label)
    return auc, acc, f1, sen, spe, pre
