import config
# from model.common.base import BaseProtoModel
from comp.nn.fusion.plain import SeqOnlyFusion, ImgOnlyFusion, CatFusion, AddFusion


def buildFusion(model,
                params_params: config.ParamsConfig=None):

    if params_params is None:
        params_params = config.train

    return fusionSwitch[params_params.Fusion['type']](model, params_params)


def _seq(model, train_params: config.ParamsConfig):
    model.FusedFeatureDim = model.SeqFeatureDim
    return SeqOnlyFusion()


def _img(model, train_params: config.ParamsConfig):
    model.FusedFeatureDim  = model.ImgFeatureDim
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


fusionSwitch = {
    'sequence': _seq,
    'image': _img,
    'cat': _cat,
    'add': _add
}