import torch.nn.functional as F
import torch

from utils.training import repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter

from model.common.base_multiloss_model import BaseMultiLossModel
import config
from utils.manager import PathManager
from utils.profiling import ClassProfiler


##########################################################
# 混合loss型ProtoNet
# 在普通ProtoNet的基础上，在fuse之前使用各个单独特征进行forward
# 产生loss，作为最终loss的一部分auxiliary loss帮助训练
# 标签推断还是根据fuse之后的特征进行的
##########################################################
class MLossProtoNet(BaseMultiLossModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source):
        super().__init__(model_params, path_manager, loss_func, data_source,
                         predict_type='logits')

        self.DistTemp = model_params.More['temperature']
        self.DistType = model_params.More.get('distance_type', 'euc')

    # @ClassProfiler("ProtoNet.forward")
    # def forward(self,                       # forward接受所有可能用到的参数
    #             support_seqs, support_imgs, support_lens, support_labels,
    #             query_seqs, query_imgs, query_lens, query_labels,
    #             epoch=None, metric='euc', is_test=False):
    #
    #     embedded_support_seqs, embedded_query_seqs, \
    #     embedded_support_imgs, embedded_query_imgs = self.embed(support_seqs, query_seqs,
    #                                                             support_lens, query_lens,
    #                                                             support_imgs, query_imgs)
    #
    #     # support seqs/imgs shape: [n*k, dim]
    #     # query seqs/imgs shape: [qk, dim]
    #     # 只有在训练阶段需要对每个特征分别forward
    #     if not is_test:
    #         # 在fusion之前将各个特征forward一次，每个特征产生一个loss
    #         seq_feature_forward_result = self._feature_forward(embedded_support_seqs, embedded_query_seqs, query_labels,
    #                                                            feature_name='sequence')
    #         img_feature_forward_result = self._feature_forward(embedded_query_seqs, embedded_query_imgs, query_labels,
    #                                                            feature_name='image')
    #
    #         # 平均各个特征产生的损失
    #         # seq和img分别使用各自的系数来缩放
    #         feature_loss = seq_feature_forward_result.get('loss') * self.AuxLossSeqFactor + \
    #                        img_feature_forward_result.get('loss') * self.AuxLossImgFactor
    #
    #     support_fused_features = self._fuse(embedded_support_seqs, embedded_support_imgs, fuse_dim=1)
    #     query_fused_features = self._fuse(embedded_query_seqs, embedded_query_imgs, fuse_dim=1)
    #
    #     # 将特征融合以后再次forward，产生一个fusion之后的损失和logits
    #     fuse_result = self._feature_forward(support_fused_features, query_fused_features, query_labels,
    #                                         feature_name='fusion')
    #     fuse_loss = fuse_result.get('loss')
    #     fuse_logits = fuse_result.get('logits')
    #
    #     if is_test:
    #         total_loss = fuse_loss  # 测试时，只返回fuse部分forward的损失值
    #     else:
    #         total_loss = fuse_loss + feature_loss   # 训练时，返回的损失值是特征损失+融合损失两部分
    #
    #     return {
    #         'logits': fuse_logits,       # 将fusion之后的logits作为预测结果
    #         'loss': total_loss,
    #         'predicts': None
    #     }

    def test(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward(is_test=True, *args, **kwargs)

    def _feature_forward(self, support_features, query_features,
                         support_labels, query_labels,
                         feature_name='none') -> dict:
        assert support_features is not None, f"[MLossProtoNet] {feature_name} is None, " \
                                             f"which is not allowed in multi-loss fusion models"

        k, n, qk = self.TaskParams.k, self.TaskParams.n, self.TaskParams.qk
        dim = support_features.size(1)

        # 取类均值作为prototype
        original_protos = support_features.view(n, k, dim).mean(dim=1)

        # 整型成为可比较的形状: [qk, n, dim]
        protos = repeatProtoToCompShape(original_protos, qk, n)
        rep_query = repeatQueryToCompShape(query_features, qk, n)

        similarity = protoDisAdapter(protos, rep_query, qk, n, dim,
                                     dis_type=self.DistType,
                                     temperature=self.DistTemp)

        logits = F.log_softmax(similarity, dim=1)
        return {
            "logits": logits,
            "loss": self.LossFunc(logits, query_labels),
            "predicts": None
        }

    def name(self):
        return "MLossProtoNet"
