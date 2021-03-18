import torch as t
import torchvision.models as models
import os

from utils.stat import statParamNumber

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# device = t.device('cuda:3')

resnet18 = models.resnet18().cuda()
print('test')
# resnet18 = models.densenet121(pretrained=True).cuda(device)

statParamNumber(resnet18)

support = t.randn((25, 3, 224, 224)).cuda()
query = t.randn((25, 3, 224, 224)).cuda()

sup_output = resnet18(support)
que_output = resnet18(query)
protos = sup_output.view(5, 5, -1).mean(dim=1).repeat((25, 1, 1))
que_output = que_output[:, None, :].repeat((1,5,1))

logits = ((que_output - protos)**2).sum(dim=2)
predicts = t.softmax(logits, dim=1)

labels = t.LongTensor([0]*25).cuda()
loss_func = t.nn.CrossEntropyLoss().cuda()

loss_val = loss_func(predicts, labels)
loss_val.backward()

# 适配3通道的ProtoNet模拟实现，显存消耗：2402M

