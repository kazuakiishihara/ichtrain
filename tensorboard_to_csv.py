from collections import defaultdict
import numpy as np
import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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


if __name__ == '__main__':
    dpath = './trained_model/20240617_190412'
    to_csv(dpath)
