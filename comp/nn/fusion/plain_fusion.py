import torch
from torch import nn


#####################################
# 只取序列特征的特征融合
#####################################
class SeqOnlyFusion(nn.Module):

    def __init__(self):
        super(SeqOnlyFusion, self).__init__()

    def forward(self, seq_features, img_features, **kwargs):
        return seq_features


#####################################
# 只取图像特征的特征融合
#####################################
class ImgOnlyFusion(nn.Module):

    def __init__(self):
        super(ImgOnlyFusion, self).__init__()

    def forward(self, seq_features, img_features, **kwargs):
        return img_features


#####################################
# 直接做序列特征和图像特征的维度连接的特征融合
#####################################
class CatFusion(nn.Module):

    def __init__(self):
        super(CatFusion, self).__init__()

    def forward(self, seq_features, img_features, fuse_dim=1, **kwargs):
        return torch.cat((seq_features, img_features), dim=fuse_dim)


#####################################
# 直接取序列特征和图像特征的相加值，一般需
# 要重投影
#####################################
class AddFusion(nn.Module):

    def __init__(self):
        super(AddFusion, self).__init__()

    def forward(self, seq_features, img_features, **kwargs):
        return seq_features + img_features


#####################################
# 直接取序列特征和图像特征的逐元素相乘，一般需
# 要重投影
#####################################
class ProductFusion(nn.Module):

    def __init__(self):
        super(ProductFusion, self).__init__()

    def forward(self, seq_features, img_features, **kwargs):
        return seq_features * img_features


