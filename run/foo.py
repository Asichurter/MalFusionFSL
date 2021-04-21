import torch as t
import torchvision.models as models
import os

from utils.stat import statParamNumber

# model = t.nn.Sequential(
#     t.nn.LSTM(input_size=128,
#               hidden_size=128,
#               num_layers=1,
#               batch_first=True,
#               bidirectional=True).cuda(),
#     t.nn.AdaptiveAvgPool1d((1,)),
#     t.nn.Linear(64, 5),
#     t.nn.Softmax(dim=1)
# )
#
# sup_data = t.randn((25, 64, 128)).cuda()
# sup_labels = t.zeros((25,)).cuda()
# sup_data = t.randn((25, 64, 128)).cuda()
# sup_labels = t.ones((25,)).cuda()
#
# loss_func = t.nn.CrossEntropyLoss().cuda()
#
# sup_predicts = model(sup_data)
# sup_loss = loss_func(sup_predicts, sup_labels)
#
# grads = t.autograd.grad(sup_loss, model.parameters(), create_graph=True)
#
# state_dict_backup = {
#     k: v.clone() for k,v in model.state_dict()
# }
#
# adapted_state_dict = {
#     k: v.clone() for k,v in model.state_dict()
# }




os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# device = t.device('cuda:3')

# model = models.resnet18().cuda()
model = models.resnet34(num_classes=256).cuda()
print('test')
# resnet18 = models.densenet121(pretrained=True).cuda(device)

statParamNumber(model)

support = t.randn((25, 3, 224, 224)).cuda()
query = t.randn((25, 3, 224, 224)).cuda()

sup_output = model(support)
que_output = model(query)
protos = sup_output.view(5, 5, -1).mean(dim=1).repeat((25, 1, 1))
que_output = que_output[:, None, :].repeat((1,5,1))

logits = ((que_output - protos)**2).sum(dim=2)
predicts = t.softmax(logits, dim=1)

labels = t.LongTensor([0]*25).cuda()
loss_func = t.nn.CrossEntropyLoss().cuda()

loss_val = loss_func(predicts, labels)
loss_val.backward()
#
# # 适配3通道的ProtoNet模拟实现，ResNet18,显存消耗：2402M
# # 适配3通道的ProtoNet模拟实现，ResNet34,显存消耗：3127M
# # 适配3通道的ProtoNet模拟实现，ResNet50,显存消耗：5035M

