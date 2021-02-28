from torch import nn


def CNNBlock2D(in_feature, out_feature, stride=1, kernel=3, padding=1,
               relu='relu', pool='max', flatten=None):
    layers = [nn.Conv2d(in_feature, out_feature,
                        kernel_size=kernel,
                        padding=padding,
                        stride=stride,
                        bias=False),
              nn.BatchNorm2d(out_feature)]

    if relu == 'relu' or relu == True:
        layers.append(nn.ReLU(inplace=True))
    elif relu == 'leaky':
        layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))

    if pool == 'max':
        layers.append(nn.MaxPool2d(2))
    elif pool == 'ada':
        layers.append(nn.AdaptiveMaxPool2d(1))

    if flatten:
        layers.append(nn.Flatten(start_dim=flatten))

    return nn.Sequential(*layers)