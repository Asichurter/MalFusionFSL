import torch
from torch import nn

import config
from utils.manager import PathManager
from model.common.base_embed_model import BaseEmbedModel
from comp.nn.other.CNN import CNNBlock2D
from utils.training import repeatProtoToCompShape, repeatQueryToCompShape, protoDisAdapter


class ConvProtoNet(BaseEmbedModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 k):

        super(ConvProtoNet, self).__init__(model_params, path_manager, loss_func, data_source)

        attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        attention_channels = [1, 32, 64, 1]
        attention_strides = [(1, 1), (1, 1), (k, 1)]
        attention_kernels = [(k, 1), (k, 1), (k, 1)]
        attention_relus = ['relu', 'relu', None]
        self.Induction = nn.Sequential(
            *[CNNBlock2D(attention_channels[i],
                         attention_channels[i + 1],
                         attention_strides[i],
                         attention_kernels[i],
                         attention_paddings[i],
                         attention_relus[i],
                         pool=None)
              for i in range(len(attention_channels) - 1)]
        )

    def forward(self,  # forward接受所有可能用到的参数
                support_seqs, support_imgs, support_lens, support_labels,
                query_seqs, query_imgs, query_lens, query_labels,
                epoch=None, metric='euc', return_embeddings=False):

        embedded_support_seqs, embedded_query_seqs, \
        embedded_support_imgs, embedded_query_imgs = self.embed(support_seqs, query_seqs,
                                                                support_lens, query_lens,
                                                                support_imgs, query_imgs)

        # support_fused_features seqs/imgs shape: [n, k, dim]
        # query seqs/imgs shape: [qk, dim]

        k, n, qk = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk

        # 直接使用seq和img的raw output进行fuse
        support_fused_features = self._fuse(embedded_support_seqs, embedded_support_imgs, fuse_dim=1)
        query_fused_features = self._fuse(embedded_query_seqs, embedded_query_imgs, fuse_dim=1)
        dim = support_fused_features.size(1)

        # 原型向量
        # shape: [n, dim]
        support_fused_features = support_fused_features.view(n, k, dim)
        support_fused_features = self.Induction(support_fused_features.unsqueeze(1)).squeeze()

        # 整型成为可比较的形状: [qk, n, dim]
        support_fused_features = repeatProtoToCompShape(support_fused_features, qk, n)
        query_fused_features = repeatQueryToCompShape(query_fused_features, qk, n)

        similarity = protoDisAdapter(support_fused_features, query_fused_features, qk, n, dim, dis_type='cos')
        logits = torch.log_softmax(similarity, dim=1)

        return {
            'logits': logits,
            'loss': self.LossFunc(logits, query_labels),
            'predict': None
        }

    def test(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward(*args, **kwargs)

    def name(self):
        return "ConvProtoNet"




