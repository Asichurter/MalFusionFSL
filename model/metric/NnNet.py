import torch.nn.functional as F
import torch

from model.common.base import BaseProtoModel
import config
from utils.manager import PathManager
from utils.profiling import ClassProfiler


class NnNet(BaseProtoModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func):
        super().__init__(model_params, path_manager, loss_func)

        self.DistTemp = model_params.More['temperature']

    # @ClassProfiler("NnNet.forward")
    def forward(self,                       # forward接受所有可能用到的参数
                support_seqs, support_imgs, support_lens, support_labels,
                query_seqs, query_imgs, query_lens, query_labels,
                epoch=None, metric='euc', return_embeddings=False):

        embedded_support_seqs, embedded_query_seqs, \
        embedded_support_imgs, embedded_query_imgs = self.embed(support_seqs, query_seqs,
                                                                support_lens, query_lens,
                                                                support_imgs, query_imgs)

        # support seqs/imgs shape: [n, k, dim]
        # query seqs/imgs shape: [qk, dim]

        k, n, qk = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk

        # 直接使用seq和img的raw output进行fuse
        support_fused_features = self._fuse(embedded_support_seqs, embedded_support_imgs, fuse_dim=2)
        query_fused_features = self._fuse(embedded_query_seqs, embedded_query_imgs, fuse_dim=1)
        dim = support_fused_features.size(2)

        # 原型向量
        # shape: [n, dim]
        # original_protos = support_fused_features.view(n, k, dim).mean(dim=1)

        rep_support = support_fused_features.repeat((qk, 1, 1)).view(qk, n*k, -1)
        rep_query = query_fused_features.repeat(n*k, 1, 1).transpose(0, 1)\
                                        .contiguous().view(qk, n*k, -1)

        # directly compare with support samples, instead of prototypes
        # shape: [qk, n*k, dim]->[qk, n, k, dim] -> [qk, n]
        similarity = ((rep_support - rep_query) ** 2).neg()\
                                        .view(qk, n, k, -1).sum(-1)
        similarity = torch.max(similarity, dim=2).values

        if return_embeddings:
            return support_seqs, query_seqs.view(qk, -1), rep_support, F.log_softmax(similarity, dim=1)

        logits = F.log_softmax(similarity, dim=1)
        return {
            "logits": logits,
            "loss": self.LossFunc(logits, query_labels),
            "predicts": None
        }

    def name(self):
        return "NnNet"

    def _fuse(self, seq_features, img_features, fuse_dim=1):
        return seq_features
        # return torch.cat((seq_features, img_features), dim=fuse_dim)