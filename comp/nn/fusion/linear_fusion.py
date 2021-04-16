from torch import nn
import torch
from copy import deepcopy

from builder.activation_builder import buildActivation
from builder.normalization_builder import buildNormalization


###########################################
# 基于双线性的特征融合
# 最好还是在之前经过一次重投影
###########################################
class BilinearFusion(nn.Module):

    def __init__(self, seq_dim, img_dim,
                 output_dim,
                 bili_norm_type, bili_affine, bili_non_linear,
                 bili_dropout=None,
                 **kwargs):
        super(BilinearFusion, self).__init__()
        self.Trans = nn.Bilinear(seq_dim, img_dim, output_dim, bias=False)

        if bili_dropout is None:
            self.Dropout = nn.Identity()
        else:
            self.Dropout = nn.Dropout(bili_dropout)

        self.Norm = buildNormalization(norm_name=bili_norm_type,
                                       feature_shape=output_dim,
                                       affine=bili_affine,
                                       norm_name_map={'bn': 'bn_1d',
                                                      'ln': "ln_1d"})

        self.NonLinear = buildActivation(bili_non_linear)

    def forward(self, seq_features, img_features, **kwargs):
        fused_features = self.Trans(seq_features, img_features)
        fused_features = self.Dropout(fused_features)
        fused_features = self.Norm(fused_features)

        return self.NonLinear(fused_features)


################################################
# 使用Hadamard积作为低秩约束的双线性特征融合的分解，用
# 于减少参数数量。
# 参见论文: "Hadamard Product for Low-Rank Bilinear
# Pooling"
################################################
class HdmProdBilinearFusion(nn.Module):

    def __init__(self, seq_dim, img_dim,
                 hidden_dim, output_dim,
                 bili_norm_type, bili_affine,
                 bili_non_linear,
                 bili_dropout=None,
                 **kwargs):
        super(HdmProdBilinearFusion, self).__init__()

        self.SeqTrans = nn.Linear(seq_dim, hidden_dim)
        self.ImgTrans = nn.Linear(img_dim, hidden_dim)
        self.OutTrans = nn.Linear(hidden_dim, output_dim)

        if bili_dropout is None:
            self.Dropout = nn.Identity()
        else:
            self.Dropout = nn.Dropout(bili_dropout)

        self.Norm = buildNormalization(norm_name=bili_norm_type,
                                       feature_shape=output_dim,
                                       affine=bili_affine,
                                       norm_name_map={'bn': 'bn_1d'})

        self.NonLinear = buildActivation(bili_non_linear)

    def forward(self, seq_features, img_features, **kwargs):
        prod = self.SeqTrans(seq_features) * self.ImgTrans(img_features)
        prod = torch.tanh(prod)     # 使用tanh激活Hadamard积
        prod = self.Dropout(prod)   # 目前dropout时添加在激活函数之后的
        return self.NonLinear(self.Norm(self.OutTrans(prod)))


class ResHdmProdBilinearFusion(HdmProdBilinearFusion):
    def __init__(self, seq_dim, img_dim,
                 hidden_dim,                    # 删除了output_dim这个参数
                 bili_norm_type, bili_affine,
                 bili_non_linear,
                 bili_dropout=None,
                 **kwargs):
        super_kwargs = deepcopy(kwargs)
        del super_kwargs['output_dim']        # 删除kw参数中已经存在的output_dim
        super(ResHdmProdBilinearFusion, self).__init__(seq_dim, img_dim,
                                                       hidden_dim, seq_dim+img_dim,     # 由于存在残差连接,因此输出维度是seq和img维度之和
                                                       bili_norm_type, bili_affine,
                                                       bili_non_linear,
                                                       bili_dropout,
                                                       **super_kwargs)

    def forward(self, seq_features, img_features, **kwargs):
        prod = self.SeqTrans(seq_features) * self.ImgTrans(img_features)
        prod = torch.tanh(prod)  # 使用tanh激活Hadamard积
        prod = self.Dropout(prod)  # 目前dropout时添加在激活函数之后的
        prod = self.NonLinear(self.Norm(self.OutTrans(prod)))
        cat = torch.cat((seq_features, img_features), dim=1)    # 固定为特征维度（1）进行cat
        return prod + cat
