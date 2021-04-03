import torch
from torch import nn
from torchvision import models


class ResNet18(torch.nn.Module):
    def __init__(self, reproject={}, **kwargs):
        super(ResNet18, self).__init__()
        hidden_dim = kwargs.get('hidden_dim', 1000)
        self.RealModel = models.resnet18(num_classes=hidden_dim)

        use_reproject = reproject.get('enabled', False)     # 空参数时默认不投影
        if use_reproject:
            out_dim = reproject['output_dim']
            self.Reproject = nn.Linear(hidden_dim, out_dim)
        else:
            self.Reproject = nn.Identity()

        dropout = reproject.get('dropout', None)            # dropout为空时默认不使用dropout
        if dropout is not None and use_reproject:           # 必须要使用重投影时才使用dropout
            self.Dropout = nn.Dropout(dropout)
        else:
            self.Dropout = nn.Identity()

    def forward(self, x):
        # channel adaption
        shape = x.size()
        if len(shape) == 4:
            if shape[1] == 1:
                x = x.repeat((1, 3, 1, 1))  # 适配ResNet的3通道
            elif shape[1] != 3:
                raise ValueError(f'[ResNet18] Channel must be 1 or 3, but got: {shape[1]}')
        elif len(shape) == 3:
            x = x[:, None, :, :].repeat(1, 3, 1, 1)

        out = self.RealModel(x)
        return self.Dropout(self.Reproject(out))
