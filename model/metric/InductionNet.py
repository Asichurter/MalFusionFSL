import torch as t
from torch import nn

import config
from utils.manager import PathManager
from model.common.base_embed_model import BaseEmbedModel
from comp.nn.relation.ntn import NTN
from utils.training import dynamicRouting, repeatProtoToCompShape, repeatQueryToCompShape


class InductionNet(BaseEmbedModel):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 ntn_hidden_size=32,
                 dynamic_routing_iter=3):
        super(InductionNet, self).__init__(model_params, path_manager, loss_func, data_source)

        self.Transformer = nn.Linear(self.FusedFeatureDim, self.FusedFeatureDim)
        self.NTN = NTN(self.FusedFeatureDim,
                       self.FusedFeatureDim,
                       ntn_hidden_size)

        self.DynamicRoutingIter = dynamic_routing_iter

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

        support_fused_features = support_fused_features.view(n, k, dim)

        # coupling shape: [n, d]
        coupling = t.zeros_like(support_fused_features).sum(dim=2)
        proto = None
        # 使用动态路由来计算原型向量
        for i in range(self.DynamicRoutingIter):
            coupling, proto = dynamicRouting(self.Transformer,
                                             support_fused_features, coupling,
                                             k)

        support_fused_features = repeatProtoToCompShape(proto, qk, n)
        query_fused_features = repeatQueryToCompShape(query_fused_features, qk, n)

        logits = self.NTN(support_fused_features, query_fused_features).view(qk, n)
        return {
            "logits": logits,
            "loss": self.LossFunc(logits, query_labels),
            "predict": None
        }

    def test(self, *args, **kwargs):
        with t.no_grad():
            return self.forward(*args, **kwargs)

    def name(self):
        return "InductionNet"




