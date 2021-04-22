import torch
from torch import nn

import config
from model.common.base_embed_model import BaseEmbedModel
from utils.manager import PathManager


class BaseMultiLossModel(BaseEmbedModel):
    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source,
                 predict_type='logits'):
        super().__init__(model_params, path_manager, loss_func, data_source)

        self.AuxLossSeqFactor = model_params.More.get('aux_loss_seq_factor', None)
        self.AuxLossImgFactor = model_params.More.get('aux_loss_img_factor', None)

        # 向下适配：如果没有分别指定seq和img的系数，则读取一个公共的系数来同时指定两个系数
        if self.AuxLossSeqFactor is None:
            self.AuxLossSeqFactor = model_params.More.get('aux_loss_factor', 0.2)
        if self.AuxLossImgFactor is None:
            self.AuxLossImgFactor = model_params.More.get('aux_loss_factor', 0.2)

        # 辅助损失系数是否为可学习的参数
        if model_params.More.get('aux_loss_learnable', False):
            self.AuxLossSeqFactor = nn.Parameter(torch.FloatTensor([self.AuxLossSeqFactor]))
            self.AuxLossImgFactor = nn.Parameter(torch.FloatTensor([self.AuxLossImgFactor]))

        assert predict_type in ['logits', 'labels'], f"[BaseMultiLossModel] Unsupported predict type: {predict_type}"
        self.PredictType = predict_type

    # 单独一个特征进行forward产生损失和标签（或logits）
    def _feature_forward(self, support_features, query_features,
                         support_labels, query_labels,
                         feature_name='none') -> dict:
        raise NotImplementedError

    def forward(self,                       # forward接受所有可能用到的参数
                support_seqs, support_imgs, support_lens, support_labels,
                query_seqs, query_imgs, query_lens, query_labels,
                epoch=None, is_test=False):

        embedded_support_seqs, embedded_query_seqs, \
        embedded_support_imgs, embedded_query_imgs = self.embed(support_seqs, query_seqs,
                                                                support_lens, query_lens,
                                                                support_imgs, query_imgs)

        # 在fusion之前将各个特征forward一次，每个特征产生一个loss
        # 只有在训练时才需要对每个特征进行forward
        if not is_test:
            seq_feature_forward_result = self._feature_forward(embedded_support_seqs, embedded_query_seqs,
                                                               support_labels, query_labels,
                                                               feature_name='sequence')
            img_feature_forward_result = self._feature_forward(embedded_support_imgs, embedded_query_imgs,
                                                               support_labels, query_labels,
                                                               feature_name='image')

            # 平均各个特征产生的损失
            # seq和img分别使用各自的系数来缩放
            feature_loss = seq_feature_forward_result.get('loss') * self.AuxLossSeqFactor + \
                           img_feature_forward_result.get('loss') * self.AuxLossImgFactor

        support_fused_features = self._fuse(embedded_support_seqs, embedded_support_imgs, fuse_dim=1)
        query_fused_features = self._fuse(embedded_query_seqs, embedded_query_imgs, fuse_dim=1)

        # 将特征融合以后再次forward，产生一个fusion之后的损失和logits
        fuse_result = self._feature_forward(support_fused_features, query_fused_features,
                                            support_labels, query_labels,
                                            feature_name='fusion')

        fuse_loss = fuse_result.get('loss')
        if is_test:
            total_loss = fuse_loss
        else:
            total_loss = fuse_loss + feature_loss

        if self.PredictType == 'labels':
            fuse_labels = fuse_result.get('predicts')
            return {
                'logits': None,
                'loss': total_loss,         # 返回的损失值是特征损失+融合损失两部分
                'predicts': fuse_labels      # 将fusion之后的predicts作为预测结果
            }
        else:
            fuse_logits = fuse_result.get('logits')
            return {
                'logits': fuse_logits,  # 将fusion之后的logits作为结果返回
                'loss': total_loss,     # 返回的损失值是特征损失+融合损失两部分
                'predicts': None
            }