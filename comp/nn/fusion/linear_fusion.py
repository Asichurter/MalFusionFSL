from torch import nn
import torch


###########################################
# 基于双线性的特征融合
# 最好还是在之前经过一次重投影
###########################################
class BilinearFusion(nn.Module):

    def __init__(self, seq_dim, img_dim, output_dim=256, bili_norm_type='bn', bili_affine=False):
        super(BilinearFusion, self).__init__()
        self.Trans = nn.Bilinear(seq_dim, img_dim, output_dim, bias=False)

        # 使用一个标准化层来约束输出，防止输出值过大
        if bili_norm_type == 'bn':
            self.Norm = nn.BatchNorm1d(output_dim, affine=bili_affine)
        elif bili_norm_type == 'ln':
            # LayerNorm的标准化范围只有输出维度
            self.Norm = nn.LayerNorm(output_dim, elementwise_affine=bili_affine)
        else:
            raise ValueError(f'[BilinearFusion] Unrecognized normalization type: {bili_norm_type}')

    def forward(self, seq_features, img_features, **kwargs):
        fused_features = self.Trans(seq_features, img_features)
        fused_features = self.Norm(fused_features)

        return fused_features


################################################
# 使用Hadamard积作为低秩约束的双线性特征融合的分解，用
# 于减少参数数量。
# 参见论文: "Hadamard Product for Low-Rank Bilinear
# Pooling"
################################################
class HdmProdBilinearFusion(nn.Module):

    def __init__(self, seq_dim, img_dim, hidden_dim, output_dim,
                 bili_norm_type=None, bili_affine=False):
        super(HdmProdBilinearFusion, self).__init__()

        self.SeqTrans = nn.Linear(seq_dim, hidden_dim)
        self.ImgTrans = nn.Linear(img_dim, hidden_dim)
        self.OutTrans = nn.Linear(hidden_dim, output_dim)

        if bili_norm_type is None:
            self.Norm = nn.Identity()
        elif bili_norm_type == 'bn':
            self.Norm = nn.BatchNorm1d(output_dim, affine=bili_affine)
        elif bili_norm_type == 'ln':
            self.Norm = nn.LayerNorm(output_dim, elementwise_affine=bili_affine)
        else:
            raise ValueError(f'[HdmProdBilinearFusion] Unrecognized normalization type: {bili_norm_type}')

    def forward(self, seq_features, img_features, **kwargs):
        prod = self.SeqTrans(seq_features) * self.ImgTrans(img_features)
        prod = torch.tanh(prod)     # 使用tanh激活Hadamard积
        return self.Norm(self.OutTrans(prod))



