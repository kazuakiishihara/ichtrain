from collections import defaultdict
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
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

def save_args(args, log_dir, file):
    log_file = os.path.join(log_dir, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(f'Script name: {file}\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

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

def tabulate_events(dpath):
    files = os.listdir(dpath)
    tfevents_files = []
    for file in files:
        if file.startswith("events.out.tfevents"):
            tfevents_files.append(file)

    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in tfevents_files]
    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps

def to_csv(dpath):
    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values)
    np_values = np.squeeze(np_values)
    np_values = np.transpose(np_values)
    df = pd.DataFrame(np_values, columns=tags)
    df.to_csv(os.path.join(dpath, 'model_performance.csv'))
