import torch as t
import torch.nn as nn

class StackConv2D(nn.Module):
    def __init__(self, channels, kernel_sizes, padding_sizes, strides, nonlinears,
                 global_pooling=True, **kwargs):
        super(StackConv2D, self).__init__()
        layers = [CNN2DBlock(channels[i], channels[i + 1],
                             stride=strides[1],
                             kernel=kernel_sizes[i],
                             padding=padding_sizes[i],
                             use_nonlinear=nonlinears[i])
                  for i in range(len(channels) - 1)]

        if global_pooling:
            layers.append(nn.AdaptiveMaxPool2d((1,1)))

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

