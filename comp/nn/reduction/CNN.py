from torch import nn as nn
from torch.nn import functional as F


###############################################################
# 参考HATT-ProtoNet中的Encoding实现，使用CNN在序列维度上进行卷积来
# 解码从RNN中提取到的特征
###############################################################
class CNNEncoder1D(nn.Module):
    def __init__(self,
                 num_channels,
                 kernel_sizes=[3],
                 paddings=[1],
                 relus=[True],
                 pools=['ada'],
                 bn=[True],
                 **kwargs):
        super(CNNEncoder1D, self).__init__()

        layers = [CNNBlock1D(num_channels[i], num_channels[i + 1],
                             kernel=kernel_sizes[i],
                             padding=paddings[i],
                             relu=relus[i],
                             pool=pools[i],
                             bn=bn[i])
                  for i in range(len(num_channels) - 1)]

        self.Encoder = nn.Sequential(*layers)

    def forward(self, x, lens=None, params=None, stats=None, prefix='Encoder'):
        # input shape: [batch, seq, dim] => [batch, dim(channel), seq]
        x = x.transpose(1,2).contiguous()

        if params is None:
            x = self.Encoder(x)
        else:
            for i, module in enumerate(self.Encoder):
                for j in range(len(module)):
                    if module[j]._get_name() == 'Conv1d':
                        x = F.conv1d(x, weight=params['%s.Encoder.%d.%d.weight' % (prefix, i, j)],
                                     stride=module[j].stride,
                                     padding=module[j].padding)
                    elif module[j]._get_name() == 'BatchNorm1d':
                        x = F.batch_norm(x,
                                         module[1].running_mean,
                                         module[1].running_var,
                                         params['%s.Encoder.%d.1.weight' % (prefix, i)],
                                         params['%s.Encoder.%d.1.bias' % (prefix, i)],
                                         momentum=1,
                                         training=self.training)
                    elif module[j]._get_name() == 'ReLU':
                        x = F.relu(x)
                    elif module[j]._get_name() == 'MaxPool1d':
                        x = F.max_pool1d(x, kernel_size=module[j].kernel_size)
                    elif module[j]._get_name() == 'AdaptiveMaxPool1d':
                        x = F.adaptive_max_pool1d(x, output_size=1)

        # shape: [batch, dim, seq]
        return x.transpose(1,2).contiguous().squeeze()


    # def static_forward(self, x, lens=None, params=None):
    #     x = x.transpose(1, 2).contiguous()
    #     for i,module in enumerate(self.Encoder):
    #         for j in range(len(module)):
    #             if module[j]._get_name() == 'Conv1d':
    #                 x = F.conv1d(x, weight=params['Encoder.Encoder.%d.%d.weight'%(i,j)],
    #                              stride=module[j].stride,
    #                              padding=module[j].padding)
    #             elif module[j]._get_name() == 'BatchNorm1d':
    #                 x = F.batch_norm(x,
    #                                  params['Encoder.Encoder.%d.1.running_mean' % i],
    #                                  params['Encoder.Encoder.%d.1.running_var' % i],
    #                                  momentum=1,
    #                                  training=True)
    #             elif module[j]._get_name() == 'ReLU':
    #                 x = F.relu(x)
    #             elif module[j]._get_name() == 'MaxPool1d':
    #                 x = F.max_pool1d(x, kernel_size=module[j].kernel_size)
    #             elif module[j]._get_name() == 'AdaptiveMaxPool1d':
    #                 x = F.adaptive_max_pool1d(x, output_size=1)
    #     return x.squeeze()


def CNNBlock1D(in_feature, out_feature, stride=1, kernel=3, padding=1,
             relu=True, pool='max', bn=True):
    layers = [nn.Conv1d(in_feature, out_feature,
                  kernel_size=kernel,
                  padding=padding,
                  stride=stride,
                  bias=False)]

    if bn:
        layers.append(nn.BatchNorm1d(out_feature))

    if relu:
        layers.append(nn.ReLU(inplace=True))

    if pool == 'ada':
        layers.append(nn.AdaptiveMaxPool1d(1))
    elif pool == 'max':
        layers.append(nn.MaxPool1d(2))

    return nn.Sequential(*layers)