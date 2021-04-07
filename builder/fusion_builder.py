import config
from comp.nn.fusion.plain_fusion import *
from comp.nn.fusion.linear_fusion import *
from comp.nn.fusion.attention_fusion import *
from comp.nn.fusion.dnn_fusion import *


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
    model.FusedFeatureDim = train_params.Fusion['params']['output_dim']

    return BilinearFusion(sdim, idim, **train_params.Fusion['params'])


def _hdmBilinear(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    model.FusedFeatureDim = train_params.Fusion['params']['output_dim']

    return HdmProdBilinearFusion(sdim, idim, **train_params.Fusion['params'])


def _resHdmBilinear(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    model.FusedFeatureDim = sdim + idim     # 残差连接是seq和img的cat，和双线性部分的输出

    return ResHdmProdBilinearFusion(sdim, idim, **train_params.Fusion['params'])


def _seqAttendImgCat(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    assert sdim == idim, f'使用seq_attend_img_cat作为fusion时，序列和图像的输出特征维度必须相同: ' \
                         f'seq_dim={sdim}, img_dim={idim}'
    model.FusedFeatureDim = sdim + idim

    return SeqAttendImgCatFusion(sdim, idim, **train_params.Fusion['params'])


def _seqAttendImgAttOnly(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    assert sdim == idim, f'使用seq_attend_img_att_only作为fusion时，序列和图像的输出特征维度必须相同: ' \
                         f'seq_dim={sdim}, img_dim={idim}'
    model.FusedFeatureDim = idim

    return SeqAttendImgAttOnlyFusion(sdim, idim, **train_params.Fusion['params'])


def _seqAttendImgResAttOnly(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    assert sdim == idim, f'使用seq_attend_img_att_only作为fusion时，序列和图像的输出特征维度必须相同: ' \
                         f'seq_dim={sdim}, img_dim={idim}'
    model.FusedFeatureDim = idim

    return SeqAttendImgResAttOnlyFusion(sdim, idim, **train_params.Fusion['params'])


def _dnnCat(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    model.FusedFeatureDim = train_params.Fusion['params']['dnn_hidden_dims'][-1]    # 融合后输出维度是dnn的最后一层的神经元数量

    return DNNCatFusion(input_dim=sdim+idim,
                        **train_params.Fusion['params'])


def _dnnCatRetCat(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    dnn_out_dim = train_params.Fusion['params']['dnn_hidden_dims'][-1]
    model.FusedFeatureDim = sdim + dnn_out_dim

    return DNNCatRetCatFusion(input_dim=sdim+idim,
                              **train_params.Fusion['params'])

def _dnnCatRetCatAll(model, train_params: config.ParamsConfig):
    sdim, idim = model.SeqFeatureDim, model.ImgFeatureDim
    dnn_out_dim = train_params.Fusion['params']['dnn_hidden_dims'][-1]
    model.FusedFeatureDim = sdim + idim + dnn_out_dim

    return DNNCatRetCatAllFusion(input_dim=sdim + idim,
                                 **train_params.Fusion['params'])


fusionSwitch = {
    'sequence': _seq,                                   # 只使用序列特征
    'image': _img,                                      # 只使用图像特征
    'cat': _cat,                                        # 使用序列特征和图像特征的堆叠
    'add': _add,                                        # 使用序列特征和图像特征向量之和
    'prod': _prod,                                      # 使用序列特征和图像特征向量之积
    'bili': _bilinear,                                  # 使用双线性输出融合特征,
    'hdm_bili': _hdmBilinear,                           # 使用Hadamard积的分解双线性融合,
    'res_hdm_bili': _resHdmBilinear,                    # 带有cat的残差连接的双线性Hadamard积分解,
    'seq_attend_img_cat': _seqAttendImgCat,             # 序列注意力对齐到图像，返回序列和对齐后的图像的堆叠
    'seq_attend_img_att_only': _seqAttendImgAttOnly,    # 序列注意力对齐到图像，只返回对齐后的图像,
    'seq_attend_img_res_att': _seqAttendImgResAttOnly,  # 序列注意力对齐到图像，只返回对齐后的图像的残差和
    'dnn_cat': _dnnCat,                                 # 基于特征cat的MLP特征融合
    'dnn_cat_ret_cat': _dnnCatRetCat,                   # 基于特征cat的MLP特征融合，返回seq和dnn输出的cat
    'dnn_cat_ret_cat_all': _dnnCatRetCatAll,            # 基于特征cat的MLP特征融合，返回seq，img和dnn输出的cat
}