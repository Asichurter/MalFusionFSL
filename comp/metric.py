import torch as t
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metric:
    def __init__(self, n, expand=False):
        self.n = n
        self.Expand = expand
        self.Labels = None

    def updateLabels(self, labels):
        self.Labels = labels

    def stat(self, out, is_labels=False, acc_only=True):
        n = self.n
        labels = self.Labels

        if not is_labels:
            if self.Expand:
                labels = t.argmax(labels, dim=1)
                out = t.argmax(out.view(-1,n), dim=1)

            else:
                out = t.argmax(out, dim=1)

        assert labels.size() == out.size()

        acc = accuracy_score(labels, out)
        precision = precision_score(labels, out, average='macro')
        recall = recall_score(labels, out, average='macro')
        f1 = f1_score(labels, out, average='macro')

        metrics = np.array([acc, precision, recall, f1])

        if acc_only:
            return np.array([acc])          # 兼容多尺度输出，只有acc时也是一个array
        else:
            return metrics