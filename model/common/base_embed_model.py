import torch as t
import torch.nn as nn
import numpy as np

from comp.nn.sequential.LSTM import BiLstmEncoder
from comp.nn.image.CNN import StackConv2D
from comp.nn.image.resnet import ResNet18
from comp.nn.reduction.CNN import CNNEncoder1D
from comp.nn.reduction.selfatt import BiliAttnReduction
from comp.nn.embedder.lstm_based import BaseLSTMEmbedder
import config
from utils.manager import PathManager
from builder.fusion import buildFusion
from model.common.base_model import BaseModel


class BaseProtoModel(BaseModel):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 **kwargs):
        super(BaseProtoModel, self).__init__(model_params, path_manager, loss_func, data_source, **kwargs)

        # ------------------------------------------------------------------------------------------
        # word embedding
        if 'sequence' in data_source:
            if model_params.Embedding['use_pretrained']:
                pretrained_matrix = t.Tensor(np.load(path_manager.wordEmbedding(), allow_pickle=True))
                self.Embedding = nn.Embedding.from_pretrained(pretrained_matrix, freeze=False)
            else:
                self.Embedding = nn.Embedding(model_params.Embedding['word_count'],
                                              embedding_dim=model_params.Embedding['embed_size'],
                                              padding_idx=0)
            self.EmbedDrop = nn.Dropout(model_params.Regularization['dropout'])
            self.SeqEmbedPipeline.append(lambda x, lens: self.EmbedDrop(self.Embedding(x)))

            # sequence encoding
            hidden_size = -1
            if model_params.SeqBackbone['seq_type'] == 'LSTM':
                self.SeqEncoder = BaseLSTMEmbedder(model_params, path_manager)
                self.SeqEmbedPipeline.append(lambda x, lens: self.SeqEncoder(x, lens))
                hidden_size = self.SeqEncoder.HiddenSize
            else:
                # TODO: 实现的其他方法的同时需要赋值hidden_size
                raise NotImplementedError("[ModelInit] Sequence modeling part has not been implemented " +
                                          "except for 'LSTM'")

            # re-projecting
            if model_params.FeatureDim is not None:
                self.SeqTrans = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=model_params.FeatureDim),
                                              nn.BatchNorm1d(model_params.FeatureDim))
                self.SeqFeatureDim = model_params.FeatureDim
            else:
                self.SeqTrans = nn.Identity()
                self.SeqFeatureDim = hidden_size

            self.SeqEmbedPipeline.append(lambda x, lens: self.SeqTrans(x))

        # ------------------------------------------------------------------------------------------
        if 'image' in data_source:
            if model_params.ConvBackbone['type'] == 'conv-4':
                self.ImageEmbedding = StackConv2D(**model_params.ConvBackbone['params']['conv-n'])
                self.ImgEmbedPipeline.append(lambda x: self.ImageEmbedding(x).squeeze())
                # output shape is the same as last channel number
                self.ImgFeatureDim = model_params.ConvBackbone['params']['conv-n']['channels'][-1]

            elif model_params.ConvBackbone['type'] == 'resnet18':
                self.ImageEmbedding = ResNet18()
                self.ImgEmbedPipeline.append(lambda x: self.ImageEmbedding(x))
                self.ImgFeatureDim = 1000   # default output shape of ResNet18 is 1000

            else:
                raise NotImplementedError(f'Not implemented image embedding module: {model_params.ConvBackbone["type"]}')

            if model_params.FeatureDim is not None:
                # 此处默认卷积网络输出的维度是通道数量，即每个feature_map最终都reduce为1x1
                self.ImgTrans = nn.Sequential(nn.Linear(in_features=model_params.ConvBackbone['params']['conv-n']['channels'][-1],
                                                        out_features=model_params.FeatureDim),
                                              nn.BatchNorm1d(model_params.FeatureDim))
                self.ImgFeatureDim = model_params.FeatureDim
            else:
                self.ImgTrans = nn.Identity()

            self.ImgEmbedPipeline.append(lambda x: self.ImgTrans(x))
        # ------------------------------------------------------------------------------------------

        # 需要在模型初始化之后才能调用fusion的初始化，因为需要检查dimension
        self.Fusion = buildFusion(self, model_params)

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
