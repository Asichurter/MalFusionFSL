from torch import nn
import torch

from builder.activation_builder import buildActivation

class FCProject(nn.Module):

    def __init__(self, in_dim, out_dim, non_linear=None, dropout=None, **kwargs):
        super(FCProject, self).__init__()

        self.Linear = nn.Linear(in_dim, out_dim, bias=False)
        self.Norm = nn.BatchNorm1d(out_dim, affine=False)
        self.NonLinear = buildActivation(non_linear)

        if dropout is not None:
            self.Dropout = nn.Dropout(dropout)
        else:
            self.Dropout = nn.Identity()

    def forward(self, x):
        x = self.Linear(x)
        x = self.NonLinear(x)
        x = self.Norm(x)
        x = self.Dropout(x)
        return x
