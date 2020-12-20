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

    @staticmethod
    def permute_hidden(hx, permutation):
        # type: (Tuple[Tensor, Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        if permutation is None:
            return hx
        return BiLstmEncoder.apply_permutation(hx[0], permutation), \
               BiLstmEncoder.apply_permutation(hx[1], permutation)

    @staticmethod
    def apply_permutation(tensor, permutation, dim=1):
        # type: (Tensor, Tensor, int) -> Tensor
        return tensor.index_select(dim, permutation)

    #################################################
    # 使用给定的参数进行forward
    #################################################
    def static_forward(self, x, lens, params):           # PyTorch1.4目前不支持在rnn上多次backward
        packed = isinstance(x, t.nn.utils.rnn.PackedSequence)
        if not packed and lens is not None:
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)

        x, batch_sizes, sorted_indices, unsorted_indices = x
        max_batch_size = int(batch_sizes[0])

        num_directions = 2
        zeros = t.zeros(self.Encoder.num_layers * num_directions,
                            max_batch_size, self.Encoder.hidden_size,
                            dtype=x.dtype, device=x.device)
        hx = (zeros, zeros)

        weights = [params['Encoder.Encoder.weight_ih_l0'],
                   params['Encoder.Encoder.weight_hh_l0'],
                   params['Encoder.Encoder.bias_ih_l0'],
                   params['Encoder.Encoder.bias_hh_l0'],
                   params['Encoder.Encoder.weight_ih_l0_reverse'],
                   params['Encoder.Encoder.weight_hh_l0_reverse'],
                   params['Encoder.Encoder.bias_ih_l0_reverse'],
                   params['Encoder.Encoder.bias_hh_l0_reverse']
                   ]

        result = _VF.lstm(x, batch_sizes, hx,
                          weights,
                          True,     # 是否bias
                          self.Encoder.num_layers,
                          self.Encoder.dropout,
                          self.Encoder.training,
                          self.Encoder.bidirectional)

        out, h = result[0], result[1:]

        out = PackedSequence(out, batch_sizes, sorted_indices, unsorted_indices)
        h = BiLstmEncoder.permute_hidden(h, unsorted_indices)

        if self.Attention is not None:
            out = BiliAttnReduction.static_forward(out, params)
            return out
        else:
            # TODO: 由于使用了CNN进行解码，因此还是可以返回整个序列
            out, lens = pad_packed_sequence(out, batch_first=True)
            return out