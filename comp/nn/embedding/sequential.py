import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from comp.nn.reduction.selfatt import BiliAttnReduction


class BiLstmEncoder(nn.Module):
    def __init__(self, input_size,
                 hidden_size=128,
                 layer_num=1,
                 dropout=0.1,
                 sequential=False,
                 bidirectional=True,
                 max_seq_len=200,
                 return_last_state=False,
                 **kwargs):

        super(BiLstmEncoder, self).__init__()

        self.Sequential = sequential
        self.MaxSeqLen = max_seq_len
        self.RetLastState = return_last_state

        self.Encoder = nn.LSTM(input_size=input_size,  # GRU
                               hidden_size=hidden_size,
                               num_layers=layer_num,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)

        # if self.SelfAtt is not None and self.SelfAtt['enabled']:
        #     if self.SelfAtt['self_att_type'] == 'custom':
        #         self.Attention = BiliAttnReduction((1 + bidirectional) * hidden_size, **self.SelfAtt['self_att_params'])
        #     else:
        #         # TODO: 可以添加一些例如多头注意力的reduction
        #         raise NotImplementedError("%s type self attention not implemented yet")
        # else:
        self.Attention = None

    def forward(self, x, lens):
        if not isinstance(x, t.nn.utils.rnn.PackedSequence) and lens is not None:
            # 由于collect的时候并没有按照长度顺序进行排序，因此此处需要设置enfore_sorted为False
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        # 进入lstm前先pack
        # x shape: [batch, seq, feature]
        # out shape: [batch, seq, 2*hidden]
        out, h = self.Encoder(x)

        # return shape: [batch, feature]
        if self.Attention is not None:
            out = self.Attention(out)

        else:
            # 由于使用了CNN进行解码，因此还是可以返回整个序列
            out, lens = pad_packed_sequence(out, batch_first=True)

            if self.RetLastState:
                out = out[:,-1,:].squeeze()

            # 如果序列中没有长度等于最大长度的元素,则使用原生pad时会产生尺寸错误
            # 此处利用0，将不足最大长度的部分填充起来
            if out.size(1) != self.MaxSeqLen:
                pad_size = self.MaxSeqLen-out.size(1)
                zero_paddings = t.zeros((out.size(0),pad_size,out.size(2))).cuda()
                out = t.cat((out,zero_paddings),dim=1)

        if self.Sequential:
            return out, lens
        else:
            return out