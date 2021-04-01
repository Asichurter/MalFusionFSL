import torch.nn.functional as F
import torch

from utils.training import repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter

from model.common.base_embed_model import BaseProtoModel
import config
from utils.manager import PathManager
from utils.profiling import ClassProfiler


class ProtoNet(BaseProtoModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source):
        super().__init__(model_params, path_manager, loss_func, data_source)

        self.DistTemp = model_params.More['temperature']

    # @ClassProfiler("ProtoNet.forward")
    def forward(self,                       # forward接受所有可能用到的参数
                support_seqs, support_imgs, support_lens, support_labels,
                query_seqs, query_imgs, query_lens, query_labels,
                epoch=None, metric='euc', return_embeddings=False):

        embedded_support_seqs, embedded_query_seqs, \
        embedded_support_imgs, embedded_query_imgs = self.embed(support_seqs, query_seqs,
                                                                support_lens, query_lens,
                                                                support_imgs, query_imgs)

        # support seqs/imgs shape: [n*k, dim]
        # query seqs/imgs shape: [qk, dim]

        k, n, qk = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk

        # 直接使用seq和img的raw output进行fuse
        support_fused_features = self._fuse(embedded_support_seqs, embedded_support_imgs, fuse_dim=1)
        query_fused_features = self._fuse(embedded_query_seqs, embedded_query_imgs, fuse_dim=1)
        dim = support_fused_features.size(1)

        # 原型向量
        # shape: [n, dim]
        original_protos = support_fused_features.view(n, k, dim).mean(dim=1)

        # 整型成为可比较的形状: [qk, n, dim]
        protos = repeatProtoToCompShape(original_protos, qk, n)
        rep_query = repeatQueryToCompShape(query_fused_features, qk, n)

        similarity = protoDisAdapter(protos, rep_query, qk, n, dim,
                                     dis_type=metric,
                                     temperature=self.DistTemp)

        if return_embeddings:
            return support_seqs, query_seqs.view(qk, -1), original_protos, F.log_softmax(similarity, dim=1)

        logits = F.log_softmax(similarity, dim=1)
        return {
            "logits": logits,
            "loss": self.LossFunc(logits, query_labels),
            "predicts": None
        }

    def test(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward(*args, **kwargs)

    def name(self):
        return "ProtoNet"