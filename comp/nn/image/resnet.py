import torch
from torchvision import models


class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.RealModel = models.resnet18()

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

        return self.RealModel(x)
