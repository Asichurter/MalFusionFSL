import config
# from model.common.base import BaseProtoModel
from comp.nn.fusion.plain_fusion import *
from comp.nn.fusion.linear_fusion import *


def buildFusion(model,
                params_params: config.ParamsConfig=None):

    if params_params is None:
        params_params = config.train

    return fusionSwitch[params_params.Fusion['type']](model, params_params)


def _seq(model, train_params: config.ParamsConfig):
    model.FusedFeatureDim = model.SeqFeatureDim
    return SeqOnlyFusion()


def _img(model, train_params: config.ParamsConfig):
    model.FusedFeatureDim = model.ImgFeatureDim
    return ImgOnlyFusion()


def _cat(model, train_params: config.ParamsConfig):
    model.FusedFeatureDim = model.SeqFeatureDim + model.ImgFeatureDim
    return CatFusion()


def _add(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    assert sdim == idim, f'使用add作为fusion时，序列和图像的输出特征维度必须相同: ' \
                         f'seq_dim={sdim}, img_dim={idim}'
    model.FusedFeatureDim = sdim
    return AddFusion()


def _prod(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    assert sdim == idim, f'使用product作为fusion时，序列和图像的输出特征维度必须相同: ' \
                         f'seq_dim={sdim}, img_dim={idim}'
    model.FusedFeatureDim = sdim
    return ProductFusion()


def _bilinear(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim

    output_dim = train_params.Fusion['params']['output_dim']
    # 部分之前完成的version中没有norm_type参数，运行时手动修改config适配一下
    normalization_type = train_params.Fusion['params']['bili_norm_type']
    use_affine = train_params.Fusion['params']['bili_affine']

    model.FusedFeatureDim = output_dim
    return BilinearFusion(sdim, idim, output_dim, normalization_type, use_affine)


def _hdmBilinear(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim

    output_dim = train_params.Fusion['params']['output_dim']
    hidden_dim = train_params.Fusion['params']['hidden_dim']
    normalization_type = train_params.Fusion['params']['bili_norm_type']
    use_affine = train_params.Fusion['params']['bili_affine']

    model.FusedFeatureDim = output_dim
    return HdmProdBilinearFusion(sdim, idim, hidden_dim, output_dim,
                                 normalization_type, use_affine)



fusionSwitch = {
    'sequence': _seq,           # 只使用序列特征
    'image': _img,              # 只使用图像特征
    'cat': _cat,                # 使用序列特征和图像特征的堆叠
    'add': _add,                # 使用序列特征和图像特征向量之和
    'prod': _prod,              # 使用序列特征和图像特征向量之积
    'bili': _bilinear,          # 使用双线性输出融合特征,
    "hdm_bili": _hdmBilinear,   # 使用Hadamard积的分解双线性融合
}