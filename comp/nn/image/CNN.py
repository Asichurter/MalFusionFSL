import torch as t
import torch.nn as nn


class ImgPatchConverter(nn.Module):
    def __init__(self):
        super(ImgPatchConverter, self).__init__()

    def forward(self, x):
        x = t.flatten(x, start_dim=2)
        x = t.transpose(x, 1, 2).contiguous()
        return x


class StackConv2D(nn.Module):
    def __init__(self, channels, kernel_sizes, padding_sizes, strides, nonlinears,
                 global_pooling=True, out_type='flatten', input_size=None,
                 global_pooling_type='max',
                 **kwargs):
        super(StackConv2D, self).__init__()
        layers = [CNN2DBlock(channels[i], channels[i + 1],
                             stride=strides[i],
                             kernel=kernel_sizes[i],
                             padding=padding_sizes[i],
                             use_nonlinear=nonlinears[i])
                  for i in range(len(channels) - 1)]

        if global_pooling:
            if global_pooling_type == 'max':
                layers.append(nn.AdaptiveMaxPool2d((1,1)))
            elif global_pooling_type == 'avg':
                layers.append(nn.AdaptiveAvgPool2d((1,1)))
            else:
                raise ValueError(f"[StackConv2D] Unrecognized global pooling type: {global_pooling_type}")
            self.OutputSize = channels[-1]

        # 将通道，图像宽高展开
        elif out_type == 'flatten':
            # shape:
            # [batch, channel, width, height]
            #               ↓
            # [batch, feature(c*w*h)]
            layers.append(nn.Flatten(1))

            out_width = input_size
            for i in range(len(strides)):
                out_width //= 2             # max_pool
                out_width //= strides[i]    # stride

            self.OutputSize = channels[-1] * out_width * out_width

        # 将图像重整为图像patch特征向量的形状
        elif out_type == 'patch':
            # shape:
            # [batch, channel, width, height]
            #               ↓
            # [batch, patch(w*h), feature(channel)]
            layers.append(ImgPatchConverter())
            self.OutputSize = channels[-1]
        else:
            raise ValueError(f'[StackConv2D] Not supported out type: {out_type}')

        self.Layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.Layers(x)

def CNN2DBlock(in_feature, out_feature, stride=1, kernel=3, padding=1, use_nonlinear=True):
    layers = [
        nn.Conv2d(in_feature, out_feature, kernel_size=kernel, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    ]
    if not use_nonlinear:
        layers.pop(2)
    return nn.Sequential(*layers)

