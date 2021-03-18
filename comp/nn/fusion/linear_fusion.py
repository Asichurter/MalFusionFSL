from torch import nn


###########################################
# 基于双线性的特征融合
# 最好还是在之前经过一次重投影
###########################################
class BilinearFusion(nn.Module):
    def __init__(self, seq_dim, img_dim, output_dim=256):
        super(BilinearFusion, self).__init__()
        self.Trans = nn.Bilinear(seq_dim, img_dim, output_dim, bias=False)
        # 使用一个标准化层来约束输出，防止输出值过大
        self.Norm = nn.BatchNorm1d(output_dim)

    def forward(self, seq_features, img_features, **kwargs):
        fused_features = self.Trans(seq_features, img_features)
        return_size = fused_features.size()
        feature_dim = return_size[-1]
        # Norm的输入必须是[batch, feature]形状的，多余的维度需要被合并
        fused_features = self.Norm(fused_features.view(-1, feature_dim))
        return fused_features.view(return_size)
