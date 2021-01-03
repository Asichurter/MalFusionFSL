import torch as t
import torch.nn as nn
import numpy as np

from comp.nn.embedding.sequential import BiLstmEncoder
from comp.nn.image.CNN import StackConv2D
from comp.nn.reduction.CNN import CNNEncoder1D
from comp.nn.reduction.selfatt import BiliAttnReduction
import config
from utils.manager import PathManager

class BaseProtoModel(nn.Module):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func):
        super(BaseProtoModel, self).__init__()

        self.LossFunc = loss_func
        self.ModelParams = model_params
        self.TaskParams = None
        self.ImageW = None

        # ------------------------------------------------------------------------------------------
        if model_params.Embedding['use_pretrained']:
            pretrained_matrix = t.Tensor(np.load(path_manager.wordEmbedding(), allow_pickle=True))
            self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
        else:
            self.Embedding = nn.Embedding(model_params.Embedding['word_count'],
                                          embedding_dim=model_params.Embedding['embed_size'],
                                          padding_idx=0)
        self.EmbedDrop = nn.Dropout(model_params.Regularization['dropout'])

        hidden_size = -1
        if model_params.SeqBackbone['seq_type'] == 'LSTM':
            self.SeqEncoder = BiLstmEncoder(input_size=model_params.Embedding['embed_size'],
                                            max_seq_len=model_params.SeqBackbone['max_seq_len'],
                                            **model_params.SeqBackbone['LSTM'])
            hidden_size = (1 + model_params.SeqBackbone['LSTM']['bidirectional']) \
                          * model_params.SeqBackbone['LSTM']['hidden_size']
        else:
            # TODO: 实现的其他方法的同时需要赋值hidden_size
            raise NotImplementedError("[ModelInit] Sequence modeling part has not been implemented except for 'LSTM'")

        # 序列约减方法按照顺序进行判断，先是自注意力，然后是时序卷积
        if model_params.SeqBackbone['self_attention']['enabled']:
            if model_params.SeqBackbone['self_attention']['type'] == 'custom':
                self.SeqReduction = BiliAttnReduction(input_dim=hidden_size,
                                                  max_seq_len=model_params.SeqBackbone['max_seq_len'])
            else:
                raise NotImplementedError("[ModelInit] Self-attention part has not been implemented except for 'custom'")
        elif model_params.SeqBackbone['temporal_conv']['enabled']:
            self.SeqReduction = CNNEncoder1D(num_channels=[hidden_size,hidden_size],
                                             **model_params.SeqBackbone['temporal_conv']['params'])
        else:
            raise NotImplementedError("[ModelInit] Self-attention part has not been implemented except for 'self-att' and 'temporal_conv'")
        # ------------------------------------------------------------------------------------------

        if model_params.ConvBackbone['type'] == 'conv-4':
            self.ImageEmbedding = StackConv2D(**model_params.ConvBackbone['params']['conv-n'])


    def _seqEmbed(self, x, lens=None):
        x = self.EmbedDrop(self.Embedding(x))
        x = self.SeqEncoder(x, lens=lens)
        x = self.SeqReduction(x, lens=lens)
        return x

    def _imgEmbed(self, x):
        return self.ImageEmbedding(x)

    def _extractEpisodeTaskStruct(self,
                                  support_seqs, query_seqs,
                                  support_imgs, query_imgs):
        # TODO: 支持更多task的输入类型来提取任务结构参数
        k = support_seqs.size(0)
        n = support_seqs.size(1)
        qk = query_seqs.size(0)

        # support img shape: [n, k, 1, w, w]
        # query img shape: [qk, 1, w, w]
        w = query_imgs.size(2)

        self.TaskParams = config.EpisodeTaskConfig(k, n, qk)
        self.ImageW = w

    def embed(self,
              support_seqs, query_seqs,
              support_lens, query_lens,
              support_imgs, query_imgs):

        self._extractEpisodeTaskStruct(support_seqs, query_seqs,
                                       support_imgs, query_imgs)

        k, n, qk, w = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk, self.ImageW

        support_seqs = support_seqs.view(n*k, -1)
        support_imgs = support_imgs.view(n*k, 1, w, w)      # 默认为单通道图片

        support_seqs = self._seqEmbed(support_seqs, support_lens).view(n, k, -1)
        query_seqs = self._seqEmbed(query_seqs, query_lens)

        support_imgs = self._imgEmbed(support_imgs).view(n, k, -1)
        query_imgs = self._imgEmbed(query_imgs).squeeze()

        assert support_seqs.size(2) == query_seqs.size(1), \
            "[BaseProtoModel.Embed] Support/query sequences' feature dimension size must match: (%d,%d)"\
            %(support_seqs.size(2),query_seqs.size(1))

        assert support_imgs.size(2) == query_imgs.size(1), \
            "[BaseProtoModel.Embed] Support/query images' feature dimension size must match: (%d,%d)"\
            %(support_imgs.size(2),query_imgs.size(1))

        return support_seqs, query_seqs, support_imgs, query_imgs

    def forward(self,                       # forward接受所有可能用到的参数
                support_seqs, support_imgs, support_lens, support_labels,
                query_seqs, query_imgs, query_lens, query_labels,
                loss_func,
                **kwargs):
        raise NotImplementedError

    def name(self):
        return "BaseProtoModel"

    def _fuse(self, seq_features, img_features, **kwargs):
        raise NotImplementedError
