import torch
from torch import nn
import numpy as np

import config
from utils.manager import PathManager
from comp.nn.sequential.LSTM import BiLstmEncoder
from comp.nn.reduction.selfatt import BiliAttnReduction
from comp.nn.reduction.CNN import CNNEncoder1D


class BaseLSTMEmbedder(nn.Module):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager):
        super(BaseLSTMEmbedder, self).__init__()

        self.LSTM = BiLstmEncoder(input_size=model_params.Embedding['embed_size'],
                                  max_seq_len=model_params.SeqBackbone['max_seq_len'],
                                  **model_params.SeqBackbone['LSTM'])
        hidden_size = (1 + model_params.SeqBackbone['LSTM']['bidirectional']) \
                      * model_params.SeqBackbone['LSTM']['hidden_size']

        # 序列约减方法按照顺序进行判断，先是自注意力，然后是时序卷积
        if model_params.SeqBackbone['LSTM']['modules']['self_attention']['enabled']:
            if model_params.SeqBackbone['LSTM']['modules']['self_attention']['type'] == 'custom':
                self.SeqReduction = BiliAttnReduction(input_dim=hidden_size,
                                                      max_seq_len=model_params.SeqBackbone['max_seq_len'])
            else:
                raise NotImplementedError(
                    "[ModelInit] Self-attention part has not been implemented except for 'custom'")

        elif model_params.SeqBackbone['LSTM']['modules']['temporal_conv']['enabled']:
            self.SeqReduction = CNNEncoder1D(num_channels=[hidden_size, hidden_size],
                                             **model_params.SeqBackbone['LSTM']['modules']['temporal_conv']['params'])

        else:
            raise NotImplementedError("[ModelInit] Self-attention part has not been implemented " +
                                      "except for 'self-att' and 'temporal_conv'")

        self.HiddenSize = hidden_size

    def forward(self, x, lens=None):
        x = self.LSTM(x, lens)
        x = self.SeqReduction(x, lens)
        return x
