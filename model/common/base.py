import torch as t
import torch.nn as nn
import numpy as np

from comp.nn.embedding.sequential import BiLstmEncoder
from comp.nn.image.CNN import StackConv2D
from comp.nn.reduction.CNN import CNNEncoder1D
from comp.nn.reduction.selfatt import BiliAttnReduction
import config
from utils.manager import PathManager
from builder.fusion import buildFusion


class BaseProtoModel(nn.Module):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 **kwargs):
        super(BaseProtoModel, self).__init__()

        self.LossFunc = loss_func
        self.ModelParams = model_params
        self.TaskParams = None
        self.ImageW = None
        self.TaskType = ""
        self.DataSource = data_source

        # ------------------------------------------------------------------------------------------
        if 'sequence' in data_source:
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

            if model_params.FeatureDim is not None:
                self.SeqTrans = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=model_params.FeatureDim),
                                              nn.BatchNorm1d(model_params.FeatureDim))
                self.SeqFeatureDim = model_params.FeatureDim
            else:
                self.SeqTrans = nn.Identity()
                self.SeqFeatureDim = hidden_size

        # ------------------------------------------------------------------------------------------
        if 'image' in data_source:
            if model_params.ConvBackbone['type'] == 'conv-4':
                self.ImageEmbedding = StackConv2D(**model_params.ConvBackbone['params']['conv-n'])

            if model_params.FeatureDim is not None:
                # 此处默认卷积网络输出的维度是通道数量，即每个feature_map最终都reduce为1x1
                self.ImgTrans = nn.Sequential(nn.Linear(in_features=model_params.ConvBackbone['params']['conv-n']['channels'][-1],
                                                        out_features=model_params.FeatureDim),
                                              nn.BatchNorm1d(model_params.FeatureDim))
                self.ImgFeatureDim = model_params.FeatureDim
            else:
                self.ImgTrans = nn.Identity()
                self.ImgFeatureDim = model_params.ConvBackbone['params']['conv-n']['channels'][-1]

        # ------------------------------------------------------------------------------------------

        self.FusedFeatureDim = None
        self.Fusion = buildFusion(self, model_params)

    def _seqEmbed(self, x, lens=None):
        x = self.EmbedDrop(self.Embedding(x))
        x = self.SeqEncoder(x, lens=lens)
        x = self.SeqReduction(x, lens=lens)
        x = self.SeqTrans(x)
        return x

    def _imgEmbed(self, x):
        x = self.ImageEmbedding(x).squeeze()
        x = self.ImgTrans(x)
        return x

    def _extractEpisodeTaskStruct(self,
                                  support_seqs, query_seqs,
                                  support_imgs, query_imgs):
        assert (support_seqs is None) ^ (query_seqs is not None), \
            f"[extractEpisodeTaskStruct] 支持集和查询集的序列数据存在性不一致: support: {support_seqs is None}, query:{query_seqs is None}"
        assert (support_imgs is None) ^ (query_imgs is not None), \
            f"[extractEpisodeTaskStruct] 支持集和查询集的图像数据存在性不一致: support: {support_imgs is None}, query:{query_imgs is None}"

        # TODO: 支持更多task的输入类型来提取任务结构参数
        if support_seqs is not None:
            k = support_seqs.size(1)
            n = support_seqs.size(0)
        elif support_imgs is not None:
            k = support_imgs.size(1)
            n = support_imgs.size(0)
        else:
            assert False, "[extractEpisodeTaskStruct] 序列和图像的支持集都为None，无法提取n,k"

        if query_seqs is not None:
            qk = query_seqs.size(0)
        elif query_imgs is not None:
            qk = query_imgs.size(0)
        else:
            assert False, "[extractEpisodeTaskStruct] 序列和图像的c查询集都为None，无法提取qk"

        # support img shape: [n, k, 1, w, w]
        # query img shape: [qk, 1, w, w]
        if support_imgs is not None:
            w = support_imgs.size(3)
        elif query_imgs is not None:
            w = query_imgs.size(2)
        else:
            w = None

        self.TaskParams = config.EpisodeTaskConfig(k, n, qk)
        self.ImageW = w

    def embed(self,
              support_seqs, query_seqs,
              support_lens, query_lens,
              support_imgs, query_imgs):

        self._extractEpisodeTaskStruct(support_seqs, query_seqs,
                                       support_imgs, query_imgs)

        k, n, qk, w = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk, self.ImageW

        # 提取任务结构时，已经判断过支持集和查询集的数据一致性，此处做单侧判断即可
        if support_seqs is not None:
            support_seqs = support_seqs.view(n * k, -1)
            support_seqs = self._seqEmbed(support_seqs, support_lens).view(n, k, -1)
            query_seqs = self._seqEmbed(query_seqs, query_lens)

            assert support_seqs.size(2) == query_seqs.size(1), \
                "[BaseProtoModel.Embed] Support/query sequences' feature dimension size must match: (%d,%d)" \
                % (support_seqs.size(2), query_seqs.size(1))

        # 提取任务结构时，已经判断过支持集和查询集的数据一致性，此处做单侧判断即可
        if support_imgs is not None:
            support_imgs = support_imgs.view(n*k, 1, w, w)      # 默认为单通道图片
            support_imgs = self._imgEmbed(support_imgs).view(n, k, -1)
            query_imgs = self._imgEmbed(query_imgs).squeeze()

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

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def _fuse(self, seq_features, img_features, **kwargs):
        return self.Fusion(seq_features, img_features, **kwargs)

    def train_state(self, mode=True):
        self.TaskType = "Train"
        super().train(mode)

    def validate_state(self):
        self.TaskType = "Validate"
        super().eval()

    def test_state(self):
        self.TaskType = "Test"
        super().eval()