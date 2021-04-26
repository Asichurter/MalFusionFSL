import torch as t
import torch.nn as nn


class TransformerSet(nn.Module):

    def __init__(self,
                 input_size,
                 dropout=0.5,
                 trans_head_nums=1,
                 **kwargs):
        super(TransformerSet, self).__init__()
        self.Transformer = nn.MultiheadAttention(embed_dim=input_size,
                                                 num_heads=trans_head_nums,
                                                 dropout=dropout)

        self.fc = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_size)


    def forward(self, x, lens=None):

        # reshape to [seq, batch, dim]
        x = x.transpose(0,1).contiguous()

        # for set-to-set operation, all sequence item is valid, no padding
        # dummy_lens = [x.size(1)]*x.size(0)

        # input as (query,key,value), namely self-attention
        residual, _weights = self.Transformer(x,x,x)
        # residual = self.Transformer(x, dummy_lens)

        residual = self.dropout(self.fc(residual))

        return self.layernorm(residual + x).transpose(0,1).contiguous()

