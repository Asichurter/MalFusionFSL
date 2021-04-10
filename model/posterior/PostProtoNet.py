import torch.nn.functional as F
import torch

from utils.training import repeatProtoToCompShape, \
                            repeatQueryToCompShape, \
                            protoDisAdapter

from model.common.base_potserior_model import BasePosteriorEmbedModel
import config
from utils.manager import PathManager
from utils.profiling import ClassProfiler


##########################################################
# 基于后融合（决策阶段融合）的原型ProtoNet模型
# loss值计算：取seq和img特征的平均
# label计算：取两特征分别softmax之后的概率值较大者的下标作为标签
##########################################################
class PostProtoNet(BasePosteriorEmbedModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source):
        super().__init__(model_params, path_manager, loss_func, data_source)

        self.DistTemp = model_params.More['temperature']
        self.DistType = model_params.More.get('distance_type', 'euc')

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

        # 各个特征各自forward，以产生各自的logits和loss
        seq_feature_forward_result = self._feature_forward(embedded_support_seqs, embedded_query_seqs, query_labels,
                                                           feature_name='sequence')
        img_feature_forward_result = self._feature_forward(embedded_query_seqs, embedded_query_imgs, query_labels,
                                                           feature_name='image')

        fused_loss = (seq_feature_forward_result.get('loss') + img_feature_forward_result.get('loss')) / 2

        seq_logits = seq_feature_forward_result.get('logits')
        img_logits = img_feature_forward_result.get('logits')

        fused_labels = self._posterior_decide_with_softmax_logits(seq_logits, img_logits)

        return {
            'logits': None,
            'loss': fused_loss,
            'predicts': fused_labels
        }

    def test(self, *args, **kwargs):
        with torch.no_grad():
            return self.forward(*args, **kwargs)

    def _feature_forward(self, support_features, query_features, query_labels,
                         feature_name, **kwargs):
        assert support_features is not None, f"[PostProtoNet] {feature_name} is None, " \
                                             f"which is not allowed in posterior fusion models"

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
        return "PostProtoNet"
