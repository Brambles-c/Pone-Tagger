import numpy as np
import pandas as pd


def multi_hot(tags: list, tag_names: list):
    tags = np.array([tag_names.index(tag) for tag in tags if tag in tag_names])
    encoding = np.zeros(len(tag_names))

    if len(tags):
        encoding[tags] = 1

    return encoding

def evaluate(preds, labels, tag_names):
    occurences = labels.sum(0)

    tp = ((preds == 1) & (labels == 1)).sum(0)
    tn = ((preds == 0) & (labels == 0)).sum(0)
    fp = ((preds == 1) & (labels == 0)).sum(0)
    fn = ((preds == 0) & (labels == 1)).sum(0)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)

    return pd.DataFrame({
        'tag': tag_names,
        'f1': (2 * prec * rec / (prec + rec)),
        'precision': prec,
        'recall': rec,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'count': occurences
    }).sort_values('f1', ascending=False)
