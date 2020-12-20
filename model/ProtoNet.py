import torch.nn.functional as F

from utils.training import repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter

from model.common.base import BaseProtoModel
from config import *
from utils.manager import PathManager


class ProtoNet(BaseProtoModel):
    def __init__(self,
                 model_params: ParamsConfig,
                 path_manager: PathManager):
        super(BaseProtoModel).__init__(model_params, path_manager)

        self.DistTemp = model_params.More['temperature']

    def forward(self,
                support_seqs, query_seqs,
                support_lens, query_lens,
                supprt_imgs, query_imgs,
                metric='euc', return_embeddings=False):

        support_seqs, query_seqs, \
        supprt_imgs, query_imgs = self.embed(support_seqs, query_seqs,
                                             support_lens, query_lens,
                                             supprt_imgs, query_imgs)

        # support seqs/imgs shape: [n, k, dim]
        # query seqs/imgs shape: [qk, dim]
        dim = support_seqs.size(2)
        k, n, qk = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk

        # 原型向量
        # shape: [n, dim]
        original_protos = support_seqs.view(n, k, dim).mean(dim=1)

        # 整型成为可比较的形状: [qk, n, dim]
        protos = repeatProtoToCompShape(original_protos, qk, n)
        rep_query = repeatQueryToCompShape(query_seqs, qk, n)

        similarity = protoDisAdapter(protos, rep_query, qk, n, dim,
                                     dis_type=metric,
                                     temperature=self.DistTemp)

        if return_embeddings:
            return support_seqs, query_seqs.view(qk, -1), original_protos, F.log_softmax(similarity, dim=1)

        return F.log_softmax(similarity, dim=1)



