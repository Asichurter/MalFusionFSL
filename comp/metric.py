import torch as t
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MetricSwitch = {
    'acc': lambda _labels, _outs: accuracy_score(_labels, _outs),
    'precision': lambda _labels, _outs: precision_score(_labels, _outs, average='macro'),
    'recall': lambda _labels, _outs: recall_score(_labels, _outs, average='macro'),
    'f1': lambda _labels, _outs: f1_score(_labels, _outs, average='macro')
}

class Metric:
    def __init__(self, n, expand=False):
        self.n = n
        self.Expand = expand
        self.Labels = None

    def updateLabels(self, labels):
        self.Labels = labels

    def stat(self, out, is_labels=False, metrics=['acc']):
        n = self.n
        labels = self.Labels.detach().cpu()
        out = out.detach().cpu()

        if not is_labels:
            if self.Expand:
                labels = t.argmax(labels, dim=1)
                out = t.argmax(out.view(-1,n), dim=1)

            else:
                out = t.argmax(out, dim=1)

        assert labels.size() == out.size()

        metric_list = []
        for metric in metrics:
            metric_list.append(MetricSwitch[metric](labels, out))

        return np.array(metric_list)