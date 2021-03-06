import torch
from torch import nn

from builder.activation_builder import buildActivation
from builder.normalization_builder import buildNormalization


def _dnn_block(_in_dim, _out_dim, _activation, _dropout):
    blocks = [nn.Linear(_in_dim, _out_dim)]

    activation_block = buildActivation(_activation)
    blocks.append(activation_block)

    if _dropout is not None:
        blocks.append(nn.Dropout(_dropout))

    return nn.Sequential(*blocks)


class _DNNFusion(nn.Module):
    def __init__(self, input_dim,
                 dnn_hidden_dims,
                 dnn_activations,
                 dnn_dropouts,
                 **kwargs):
        super(_DNNFusion, self).__init__()
        assert len(dnn_hidden_dims) == len(dnn_activations) == len(dnn_dropouts), \
            f'[DNNFusion] len of hidden_dims, activations and dropouts must have the same len, ' \
            f'but got {len(dnn_hidden_dims)},{len(dnn_activations)},{len(dnn_dropouts)}'

        dims = [input_dim] + dnn_hidden_dims
        layers = [
            _dnn_block(dims[i], dims[i+1], dnn_activations[i], dnn_dropouts[i])
            for i in range(len(dnn_hidden_dims))
        ]

        self.Layers = nn.Sequential(*layers)
        self._DnnOutputDim = dnn_hidden_dims[-1]

    def dnn_forward(self, x):
        return self.Layers(x)


class DNNCatFusion(_DNNFusion):
    def __init__(self, input_dim,
                 dnn_hidden_dims,
                 dnn_activations,
                 dnn_dropouts,
                 dnn_norm_type=None,
                 feature_dims={},
                 **kwargs):

        super().__init__(input_dim,
                         dnn_hidden_dims,
                         dnn_activations,
                         dnn_dropouts,
                         **kwargs)

        cat_dim = self._DnnOutputDim
        self.Norm = buildNormalization(dnn_norm_type, cat_dim,
                                       norm_name_map={'bn': 'bn_1d', 'in': 'in_1d'})

    def forward(self, seq_features, img_features, fused_dim=1, **kwargs):
        cat_features = torch.cat((seq_features, img_features), dim=fused_dim)
        return self.Norm(self.dnn_forward(cat_features))


class DNNCatRetCatFusion(_DNNFusion):
    def __init__(self, input_dim,
                 dnn_hidden_dims,
                 dnn_activations,
                 dnn_dropouts,
                 dnn_norm_type=None,
                 feature_dims={},
                 **kwargs):

        super().__init__(input_dim,
                         dnn_hidden_dims,
                         dnn_activations,
                         dnn_dropouts,
                         **kwargs)

        try:
            seq_dim = feature_dims['seq']
        except KeyError as e:
            raise ValueError(f"[DNNCatRetCatFusion] Unable to fetch sequence dim from param: {str(e)}")

        cat_dim = seq_dim + self._DnnOutputDim
        self.Norm = buildNormalization(dnn_norm_type, cat_dim,
                                       norm_name_map={'bn': 'bn_1d', 'in': 'in_1d'})

    def forward(self, seq_features, img_features, fused_dim=1, **kwargs):
        cat_features = torch.cat((seq_features, img_features), dim=fused_dim)
        cat_features = self.dnn_forward(cat_features)
        return self.Norm(torch.cat((seq_features, cat_features), dim=fused_dim))       # 返回seq和dnn输出的cat


class DNNCatRetCatAllFusion(_DNNFusion):
    def __init__(self, input_dim,
                 dnn_hidden_dims,
                 dnn_activations,
                 dnn_dropouts,
                 dnn_norm_type=None,
                 feature_dims={},
                 **kwargs):

        super().__init__(input_dim,
                         dnn_hidden_dims,
                         dnn_activations,
                         dnn_dropouts,
                         **kwargs)

        try:
            seq_dim = feature_dims['seq']
            img_dim = feature_dims['img']
        except KeyError as e:
            raise ValueError(f"[DNNCatRetCatAllFusion] Unable to fetch sequence/image dim from param: {str(e)}")

        cat_dim = seq_dim + img_dim + self._DnnOutputDim
        self.Norm = buildNormalization(dnn_norm_type, cat_dim,
                                       norm_name_map={'bn': 'bn_1d', 'in': 'in_1d'})

    def forward(self, seq_features, img_features, fused_dim=1, **kwargs):
        cat_features = torch.cat((seq_features, img_features), dim=fused_dim)
        cat_features = self.dnn_forward(cat_features)

        # 返回seq，img和dnn输出的三层cat
        return self.Norm(torch.cat((seq_features, img_features, cat_features), dim=fused_dim))


