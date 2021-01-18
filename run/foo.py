import torchvision.models as models
import os

from utils.stat import statParamNumber

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# resnet18 = models.resnet18(pretrained=True).cuda()
resnet18 = models.densenet121(pretrained=True).cuda()

statParamNumber(resnet18)