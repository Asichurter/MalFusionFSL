import torch as t
from torch import nn as nn
from torch.nn import functional as F

from utils.training import getMaskFromLens


class BiliAttnReduction(nn.Module):
    def __init__(self, input_dim, max_seq_len=200,
                 **kwargs):
        super(BiliAttnReduction, self).__init__()

        self.MaxSeqLen = max_seq_len

        self.IntAtt = nn.Linear(input_dim, input_dim, bias=False)
        self.ExtAtt = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x, lens=None):
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        feature_dim = x.size(2)

        # weight shape: [batch, seq, 1]
        att_weight = self.ExtAtt(t.tanh(self.IntAtt(x))).squeeze()

        if lens is not None:
            if not isinstance(lens, t.Tensor):
                lens = t.Tensor(lens)

            mask = getMaskFromLens(lens,self.MaxSeqLen)
            att_weight.masked_fill_(mask, float('-inf'))

        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1,1,feature_dim))
        return (att_weight * x).sum(dim=1)

    @staticmethod
    def static_forward(x, params, lens=None):               # 此处由于命名限制，假定参数是按照使用顺序feed进来的
        if isinstance(x, t.nn.utils.rnn.PackedSequence):
            x, lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        feature_dim = x.size(2)

        att_weight = F.linear(input=t.tanh(F.linear(input=x,
                                                    weight=params[0])),
                              weight=params[1]).squeeze()

        if lens is not None:
            if not isinstance(lens, t.Tensor):
                lens = t.Tensor(lens)
            mask = getMaskFromLens(lens)
            att_weight.masked_fill_(mask, float('-inf'))

        att_weight = t.softmax(att_weight, dim=1).unsqueeze(-1).repeat((1,1,feature_dim))
        return (att_weight * x).sum(dim=1)



# class
class SelfAttnReduction(nn.Module):
    def __init__(self,
                 input_size,
                 max_seq_len=200,
                 **kwargs):
        super(SelfAttnReduction, self).__init__()

        self.MaxSeqLen = max_seq_len
        self.Dim = input_size

        self.Dropout = nn.Dropout(p=kwargs['dropout'])

        self.K = nn.Linear(input_size, input_size)
        self.Q = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, input_size)

    def forward(self, x, lens=None):

        x_k = t.tanh(self.Dropout(self.K(x)))
        x_q = t.tanh(self.Dropout(self.Q(x)))
        x_v = t.tanh(self.Dropout(self.V(x)))

        w = t.bmm(x_k, x_q.permute(0,2,1)) / (self.Dim**0.5)
        w = t.sum(w, dim=-1)

        if lens is not None:
            mask = getMaskFromLens(lens,
                                   max_seq_len=self.MaxSeqLen)
            w.masked_fill_(mask, value=float('-inf'))

        # 将K与Q矩阵相乘以后得到的"序列-序列"权重在最后一个维度相加,约减为"序列"维度
        w = w.softmax(dim=1).unsqueeze(-1).expand_as(x_v)
        attended_v = (x_v * w).sum(dim=-1)

        return attended_v


























