import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config
from model.common.base_embed_model import BaseEmbedModel
from utils.manager import PathManager
from comp.nn.set2set import getSet2SetFunc


from utils.training import repeatProtoToCompShape, repeatQueryToCompShape, protoDisAdapter


class FEAT(BaseEmbedModel):

    def __init__(self,
                 model_params: config.ParamsConfig,
                 path_manager: PathManager,
                 loss_func,
                 data_source):

        super(FEAT, self).__init__(model_params, path_manager, loss_func, data_source)

        feat_params = model_params.ModelFeture['feat']
        self.Avg = feat_params['avg']
        self.ContraFac = feat_params['contrastive_factor']
        self.DisTemp = model_params.More['temperature']

        set_func_type = feat_params['set_to_set_func']
        input_size = self.FusedFeatureDim
        self.SetFunc = getSet2SetFunc(set_func_type, input_size, **feat_params)

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
        qk_per_class = qk // n

        # 直接使用seq和img的raw output进行fuse
        support_fused_features = self._fuse(embedded_support_seqs, embedded_support_imgs, fuse_dim=1)
        query_fused_features = self._fuse(embedded_query_seqs, embedded_query_imgs, fuse_dim=1)
        dim = support_fused_features.size(1)

        # contrastive-loss for regulization during training
        if self.training and self.ContraFac is not None:
            # union shape: [n, qk+k, dim]
            # here suppose query set is constructed in group by class
            union = t.cat((support_fused_features.view(n, k, dim), query_fused_features.view(n, qk_per_class, dim)),
                          dim=1)                        # TODO: make it capable to process in batch

            adapted_union = self.SetFunc(union)

            # post-avg in default
            adapted_proto = adapted_union.mean(dim=1)

            # union shape: [(qk+k)*n, dim]
            adapted_union = adapted_union.view((qk_per_class + k) * n, dim)

            # let the whole dataset execute classification task based on the adapted prototypes
            adapted_proto = repeatProtoToCompShape(adapted_proto, (qk_per_class + k) * n, n)
            adapted_union = repeatQueryToCompShape(adapted_union, (qk_per_class + k) * n, n)

            adapted_sim = protoDisAdapter(adapted_proto, adapted_union,
                                          (qk_per_class + k) * n, n, dim, dis_type='euc')

            # here, the target label set has labels for both support set and query set,
            # where labels permute in order and cluster (every 'qk_per_class+k')
            adapted_logits = F.log_softmax(adapted_sim, dim=1)

        # if return_unadapted:
        #     unada_support = support_fused_features.view(n,k,-1).mean(1)
        #     unada_support = repeatProtoToCompShape(unada_support,
        #                                            qk, n)

        ################################################################
        if self.Avg == 'post':

            # support set2set
            support_fused_features = self.SetFunc(support_fused_features.view(1,n*k,dim))

            # shape: [n, dim]
            support_fused_features = support_fused_features.view(n, k, dim).mean(dim=1)

        elif self.Avg == 'pre':

            # shape: [n, dim]
            support_fused_features = support_fused_features.view(n, k, dim).mean(dim=1)
            # support set2set
            support_fused_features = self.SetFunc(support_fused_features.unsqueeze(0))
        ################################################################


        # shape: [n, dim] -> [1, n, dim]
        # pre-avg in default, treat prototypes as sequence
        # support = support.view(n, k, dim).mean(dim=1).unsqueeze(0)
        # # support set2set
        # support = self.SetFunc(support)

        support_fused_features = repeatProtoToCompShape(support_fused_features, qk, n)
        query_fused_features = repeatQueryToCompShape(query_fused_features, qk, n)

        similarity = protoDisAdapter(support_fused_features, query_fused_features, qk, n, dim,
                                     dis_type='euc',
                                     temperature=self.DisTemp)

        logits = F.log_softmax(similarity, dim=1)
        loss = self.LossFunc(logits, query_labels)

        # 在原损失基础上添加一个对比损失值帮助训练
        if self.training and self.ContraFac is not None:
            # 此处假设没有shuffle，标签直接从0排列到n
            adapted_labels = t.arange(0, n, dtype=t.long).cuda()
            adapted_labels = adapted_labels.unsqueeze(1).expand((n,(qk_per_class+k))).flatten()
            contrastive_loss = self.LossFunc(adapted_logits, adapted_labels)
            loss += self.ContraFac * contrastive_loss

        return {
            'logits': logits,
            'loss': loss,
            'predict': None
        }

        # else:
        #     # if return_unadapted:
        #     #     unada_sim = protoDisAdapter(unada_support, query, qk, n, dim,
        #     #                          dis_type='euc',
        #     #                          temperature=self.DisTempr)
        #     #     return F.log_softmax(similarity, dim=1), F.log_softmax(unada_sim, dim=1)
        #     #
        #     # else:
        #         return F.log_softmax(similarity, dim=1)

    def test(self, *args, **kwargs):
        with t.no_grad():
            return self.forward(*args, **kwargs)

    def name(self):
        return "FEAT"

