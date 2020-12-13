import torch as t
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
import random as rd
import time

from comp.metric import Metric
from utils.magic import magicSeed, randPermutation, randomList
from comp.sampler import EpisodeSampler

#########################################
# 基于Episode训练的任务类，包含采样标签空间，
# 采样实验样本，使用dataloader批次化序列样
# 本并且将任务的标签标准化。
#
# 调用episode进行任务采样和数据构建。
# 在采样时会自动缓存labels，输入模型的输出
# 调用accuracy计算得到正确率
#########################################
class EpisodeTask:
    def __init__(self, k, qk, n, N, dataset, cuda=True,
                 label_expand=False, parallel=None):
        self.UseCuda = cuda
        self.Dataset = dataset
        self.Expand = label_expand
        self.Parallel = parallel

        assert k + qk <= N, '支持集和查询集采样总数大于了类中样本总数!'
        self.Params = {'k': k, 'qk': qk, 'n': n, 'N': N}

        self.SupSeqLenCache = None
        self.QueSeqLenCache = None

        self.LabelsCache = None
        self.Metric = Metric(n, expand=label_expand)

        self.TaskSeedCache = None
        self.SamplingSeedCache = None

    def _readParams(self):
        params = self.Params
        k, qk, n, N = params['k'], params['qk'], params['n'], params['N']

        return k, qk, n, N

    def _getLabelSpace(self, seed=None):
        seed = magicSeed() if seed is None else seed
        self.TaskSeedCache = seed

        sampled_classes = randPermutation(self.Dataset.TotalClassNum, seed)

        # print('label space: ', sampled_classes)
        return sampled_classes

    def _getTaskSampler(self, label_space, seed=None):
        sampling_seed = magicSeed() if seed is None else seed

        self.SamplingSeedCache = sampling_seed

        k, qk, n, N = self._readParams()

        seed_for_each_class = randomList(num=len(label_space),      # 采样每个class内部的seed
                                         seed=sampling_seed,
                                         allow_duplicate=True)

        support_sampler = EpisodeSampler(k, qk, N, label_space, seed_for_each_class, mode='support')
        query_sampler = EpisodeSampler(k, qk, N, label_space, seed_for_each_class, mode='query')        # make query set labels cluster

        return support_sampler, query_sampler

    def _getEpisodeData(self, support_sampler, query_sampler):
        k, qk, n, N = self._readParams()

        support_loader = DataLoader(self.Dataset, batch_size=k * n,
                                    sampler=support_sampler, collate_fn=batchSequence)#getBatchSequenceFunc())
        query_loader = DataLoader(self.Dataset, batch_size=qk * n,
                                  sampler=query_sampler, collate_fn=batchSequence)#getBatchSequenceFunc())

        support_seqs, support_imgs, support_lens, support_labels = support_loader.__iter__().next()
        query_seqs, query_imgs, query_lens, query_labels = query_loader.__iter__().next()

        # 将序列长度信息存储便于unpack
        self.SupSeqLenCache = support_lens
        self.QueSeqLenCache = query_lens
        # 更新查询集标签到metric管理器中
        self.Metric.updateLabels(query_labels)

        return (support_seqs, support_imgs, support_lens, support_labels), \
               (query_seqs, query_imgs, query_lens, query_labels)

    def _taskLabelNormalize(self, sup_labels, que_labels):
        k, qk, n, N = self._readParams()

        # 由于分类时是按照类下标与支持集进行分类的，因此先出现的就是第一类，每k个为一个类
        # size: [ql*n]
        sup_labels = sup_labels[::k].repeat(len(que_labels))  # 支持集重复q长度次，代表每个查询都与所有支持集类比较
        que_labels = que_labels.view(-1, 1).repeat((1, n)).view(-1)  # 查询集重复n次

        assert sup_labels.size(0) == que_labels.size(0), \
            '扩展后的支持集和查询集标签长度: (%d, %d) 不一致!' % (sup_labels.size(0), que_labels.size(0))

        # 如果进行扩展的话，每个查询样本的标签都会是n维的one-hot（用于MSE）
        # 不扩展是个1维的下标值（用于交叉熵）
        if not self.Expand:
            que_labels = t.argmax((sup_labels == que_labels).view(-1, n).int(), dim=1)
            return que_labels.long()
        else:
            que_labels = (que_labels == sup_labels).view(-1, n)
            return que_labels.float()

    def episode(self, task_seed=None, sampling_seed=None):
        raise NotImplementedError

    def labels(self):
        return self.LabelsCache

    def metrics(self, out, is_labels=False, acc_only=True):
        return self.Metric.stat(out, acc_only, is_labels)


##########################################################
# 常规episode任务，通用于meta-learning模型中，episode返回中
# 包含了所有可能用到的数据
##########################################################
class RegularEpisodeTask(EpisodeTask):
    def __init__(self, k, qk, n, N, dataset, cuda=True,
                 label_expand=False, parallel=None):
        super(RegularEpisodeTask, self).__init__(k, qk, n, N, dataset, cuda, label_expand, parallel)

    def episode(self, task_seed=None, sampling_seed=None):
        k, qk, n, N = self._readParams()

        label_space = self._getLabelSpace(task_seed)
        support_sampler, query_sampler = self._getTaskSampler(label_space, sampling_seed)
        (support_seqs, support_imgs, support_lens, support_labels), \
        (query_seqs, query_imgs, query_lens, query_labels) = self._getEpisodeData(support_sampler, query_sampler)
        img_width, img_height = support_imgs.size()[-2:]

        labels = self._taskLabelNormalize(support_labels, query_labels)
        self.LabelsCache = labels

        if self.UseCuda:
            support_seqs = support_seqs.cuda()
            support_imgs = support_imgs.cuda()
            query_seqs = query_seqs.cuda()
            query_imgs = query_imgs.cuda()
            labels = labels.cuda()

        # if self.Parallel is not None:
        #     supports = supports.repeat((len(self.Parallel),1,1,1))
        #     self.SupSeqLenCache = [self.SupSeqLenCache]*len(self.Parallel)

        # 重整数据结构，便于模型读取任务参数
        support_seqs = support_seqs.view(n, k, -1)
        support_imgs = support_imgs.view(n, k, 1, img_width, img_height)
        query_seqs = query_seqs.view(n*qk, -1)
        query_imgs = query_imgs.view(n*qk, 1, img_width, img_height)    # 注意，此处的qk指每个类中的查询样本个数，并非查询集长度

        return (support_seqs, support_imgs, support_lens, support_labels), \
               (query_seqs, query_imgs, query_lens, query_labels)

########################################################
# dataloader从dataset中收集数据的收集方法
# 将逐个抽出的api，img，seqlen和label进行归类收集返回
# 返回的出口在dataloader中
########################################################
def batchSequence(data):
    seqs, imgs, lens, labels = [], [], [], []

    for seq, img, len_, label in data:
        seqs.append(seq)
        imgs.append(img)
        lens.append(len_)
        labels.append(label)

    # 没有按照序列长度排序，需要在pack时设置enforce_sort = False
    return seqs, imgs, lens, labels

if __name__ == '__main__':
    t = RegularEpisodeTask()