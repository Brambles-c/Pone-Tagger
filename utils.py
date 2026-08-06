from skmultilearn.model_selection import iterative_train_test_split
import numpy as np
from torch import Tensor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from config import images_path
from PIL import Image
import os


def multi_hot(tags: list, tag_names: list):
    tags = np.array([tag_names.index(tag) for tag in tags if tag in tag_names])
    encoding = np.zeros(len(tag_names))

    if len(tags):
        encoding[tags] = 1

    return encoding


def stratified_split(tag_col: pd.Series):
    y = np.stack(tag_col.to_numpy())

    if os.path.exists('data/indices.npz'):
        with np.load('data/indices.npz') as stored:
            idx_train = stored['train']
            idx_test = stored['test']

        y_train = y[idx_train]
        y_test = y[idx_test]
    else:
        idx = tag_col.index.to_numpy().reshape(-1, 1)

        idx_train, y_train, idx_test, y_test = iterative_train_test_split(idx, y, 0.1)

        idx_train = idx_train.reshape(-1)
        idx_test = idx_test.reshape(-1)
        np.savez(
            'data/indices.npz',
            train=idx_train,
            test=idx_test,
        )
    
    return idx_train, y_train, idx_test, y_test


def confusion_matrix(preds, labels: np.ndarray, tag_names):
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


def visualize(X: Tensor, y: Tensor, preds: np.ndarray, tag_names, width=2):
    X = X.permute(0, 2, 3, 1)
    tag_names = np.array(tag_names)

    fig, axis = plt.subplots(1, len(preds), figsize=(width * len(X), width), constrained_layout=True)
    axis: list[Axes]

    for ax, img in zip(axis, X):
        ax.imshow(img)
        ax.axis(False)

    idx_sorted = preds.argsort()[:, ::-1].copy()

    preds = [
        '\n'.join([
            f'{label_name + ':  ':>20}{p:.2f}  {label}'
            for p, label_name, label in zip(
                preds[row, idx_cols],
                tag_names[idx_cols],
                y[row, idx_cols]
            )
            if p > 0.5
        ])
        for row, idx_cols in enumerate(idx_sorted)
    ]

    plt.show()
    print(f'\n\n'.join(preds))
