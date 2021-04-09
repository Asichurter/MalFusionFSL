import torch

from .base_embed_model import BaseEmbedModel

import config
from utils.manager import PathManager


#############################################
# 基于嵌入的后融合(决策融合)模型的基类
# 需要实现基于某一类特征完成forward计算loss和进行inference
# 的方法
#############################################
class BasePosteriorEmbedModel(BaseEmbedModel):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source):
        super().__init__(model_params, path_manager, loss_func, data_source, need_fusion=False)

    def _feature_forward(self, support_features, query_features,
                         query_labels,
                         feature_name,
                         **kwargs) -> dict:
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError("[BasePosteriorEmbedModel] 'test' method must be implemented by child model")

    def _posterior_decide_with_softmax_logits(self, *logits):
        '''
        使用多个feature分别forward之后生成的softmax logits进行联合决策。

        基本方法是：所有样本先在每个feature内选出一个概率值最高的结果，
        然后各个feature之间最高的概率值再进行比较来决出最高的概率值，
        并且以该最高概率值对应的标签作为联合决策结果返回
        '''
        m_logit_list = []
        m_label_list = []
        for logit in logits:
            # 对于每一个feature的softmax logits，先取出每一个样本的最大值和最大值对应的下标
            max_item = torch.max(logit, dim=1)
            m_logit_list.append(max_item.values)
            m_label_list.append(max_item.indices)

        logit_tensor = torch.vstack(m_logit_list).transpose(0,1).contiguous().cuda()
        label_tensor = torch.vstack(m_label_list).transpose(0,1).contiguous().cuda()

        # 对所有feature的逐样本最大logit，找出最大的logit对应的feature
        label_indexes = torch.max(logit_tensor, dim=1, keepdim=True).indices
        # 返回最大logit的feature的最大值的label
        ret_labels = torch.gather(label_tensor, 1, label_indexes)

        return ret_labels.squeeze()

