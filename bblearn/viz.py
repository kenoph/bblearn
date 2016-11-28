#!/usr/bin/env python

import numpy as np
from matplotlib import pylab as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import seaborn as sns
sns.set()


#TODO: Write tests :)
def confusion_matrix(ytrue, ypred, labels=None, normalize=True, percent=True, size=20, fmt='.0f'):
    """
    Render the confusion matrix using seaborn.
    Labels are taken from ytrue values if not given as parameters.
    """
    if labels is None:
        labels = sorted(np.unique(ytrue))

    cm = sk_confusion_matrix(ytrue, ypred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    if percent:
        cm = cm * 100.0

    plt.figure(figsize=(size, size))
    ax = sns.heatmap(cm, annot=True, fmt=fmt, linewidths=.5, cbar=False)
    ax.xaxis.tick_top()
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels[::-1])
    ax.set_title("Confusion Matrix", y=1.1, size="x-large", weight='bold')
    ax.set_ylabel("True Label", weight='bold')
    ax.set_xlabel("Predicted Label", weight='bold')

