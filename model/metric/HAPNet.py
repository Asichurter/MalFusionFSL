import torch as t
import torch.nn as nn
import warnings

import config
from model.common.base_embed_model import BaseEmbedModel
from comp.nn.other.CNN import CNNBlock2D
from utils.manager import PathManager


class InstanceAttention(nn.Module):
    def __init__(self, linear_in, linear_out):
        super(InstanceAttention, self).__init__()
        self.g = nn.Linear(linear_in, linear_out)

    def forward(self, support, query, k, qk, n):
        # support/query shape: [qk*n*k, d]
        d = support.size(1)
        support = self.g(support)
        query = self.g(query)
        # shape: [qk, n, k, d]->[qk, n, k]
        attentions = t.tanh((support * query).view(qk, n, k, d)).sum(dim=3).squeeze()
        # shape: [qk,n,k]->[qk,n,k,d]
        attentions = t.softmax(attentions, dim=2).unsqueeze(3).repeat(1, 1, 1, d)

        return t.mul(attentions, support.view(qk, n, k, d))


class FeatureAttention(nn.Module):
    def __init__(self, k):
        super(FeatureAttention, self).__init__()
        # if k % 2 == 0:
        #     warnings.warn("K=%d是偶数将会导致feature_attention中卷积核的宽度为偶数，因此部分将会发生一些变化")
        #     attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        # else:
        attention_paddings = [(k // 2, 0), (k // 2, 0), (0, 0)]
        attention_channels = [1, 32, 64, 1]
        attention_strides = [(1, 1), (1, 1), (k, 1)]
        attention_kernels = [(k, 1), (k, 1), (k, 1)]
        attention_relus = ['relu', 'relu', 'relu']

        self.Layers = nn.Sequential(
            *[CNNBlock2D(attention_channels[i],
                         attention_channels[i + 1],
                         attention_strides[i],
                         attention_kernels[i],
                         attention_paddings[i],
                         attention_relus[i],
                         pool='none')
              for i in range(len(attention_channels) - 1)])

    def forward(self, x):
        return self.Layers(x)


class HAPNet(BaseEmbedModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 k):
        super(HAPNet, self).__init__(model_params, path_manager, loss_func, data_source)

        # 获得样例注意力的模块
        # 将嵌入后的向量拼接成单通道矩阵后，有多少个支持集就为几个batch
        self.FeatureAttention = FeatureAttention(k)

        # 获得样例注意力的模块
        # 将support重复query次，query重复n*k次，因为每个support在每个query下嵌入都不同
        self.InstanceAttention = InstanceAttention(self.FusedFeatureDim, self.FusedFeatureDim)

    # @ClassProfiler("ProtoNet.forward")
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

        # 将嵌入的支持集展为合适形状
        # support_fused_features shape: [n,k,d]->[n,k,d]
        support_fused_features = support_fused_features.view(n, k, dim)
        # query shape: [qk, d]
        query_fused_features = query_fused_features.view(qk, -1)

        # 将支持集嵌入视为一个单通道矩阵输入到特征注意力模块中获得特征注意力
        # 并重复qk次让基于支持集的特征注意力对于qk个样本相同
        # 输入: [n,k,d]->[n,1,k,d]
        # 输出: [n,1,1,d]->[n,d]->[qk,n,d]
        feature_attentions = self.FeatureAttention(support_fused_features.unsqueeze(dim=1)).squeeze().repeat(qk, 1, 1)

        # 将支持集重复qk次，将查询集重复n*k次以获得qk*n*k长度的样本
        # 便于在获得样例注意力时，对不同的查询集有不同的样例注意力
        # 将qk，n与k均压缩到一个维度上以便输入到线性层中
        # query_expand shape:[qk,d]->[n*k,qk,d]->[qk,n,k,d]
        # support_expand shape: [n,k,d]->[qk,n,k,d]
        support_expand = support_fused_features.repeat((qk, 1, 1, 1)).view(qk * n * k, -1)
        query_expand = query_fused_features.repeat((n * k, 1, 1)).transpose(0, 1).contiguous().view(qk * n * k, -1)

        # 利用样例注意力注意力对齐支持集样本
        # shape: [qk,n,k,d]
        support_fused_features = self.InstanceAttention(support_expand, query_expand, k, qk, n)

        # 生成对于每一个qk都不同的类原型向量
        # 注意力对齐以后，将同一类内部的加权的向量相加以后
        # proto shape: [qk,n,k,d]->[qk,n,d]
        support_fused_features = support_fused_features.sum(dim=2).squeeze()
        # support_fused_features = support_fused_features.mean(dim=1).repeat((qk,1,1)).view(qk,n,-1)

        # query shape: [qk,d]->[qk,n,d]
        query_fused_features = query_fused_features.unsqueeze(dim=1).repeat(1, n, 1)

        # dis_attented shape: [qk*n,n,d]->[qk*n,n,d]->[qk*n,n]
        # dis_attented = (((support_fused_features-query)**2)).sum(dim=2).neg()
        dis_attented = (((support_fused_features - query_fused_features) ** 2) * feature_attentions).sum(dim=2).neg()

        logits = t.log_softmax(dis_attented, dim=1)
        return {
            "logits": logits,
            "loss": self.LossFunc(logits, query_labels),
            "predicts": None
        }

    def test(self, *args, **kwargs):
        with t.no_grad():
            return self.forward(*args, **kwargs)

    def name(self):
        return "HAPNet"
