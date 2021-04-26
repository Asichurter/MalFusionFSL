import torch
from torch import nn


#####################################################
# Neural Tensor Layer, 用于建模向量关系，实质是一个双线性层
#####################################################
class NTN(nn.Module):
    def __init__(self, c, e, k):
        super(NTN, self).__init__()
        self.Bilinear = nn.Bilinear(c, e, k, bias=False)
        self.Scoring = nn.Sequential(
            nn.ReLU(),
            nn.Linear(k, 1, bias=True),
        )

    def forward(self, c, e):
        v = self.Bilinear(c, e)
        s = self.Scoring(v)
        s = torch.sigmoid(s)
        return s
