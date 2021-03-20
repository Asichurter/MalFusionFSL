import random as rd
from torch.utils.data.sampler import Sampler

from utils.magic import randPermutation, magicSeed


class EpisodeSampler(Sampler):

    def __init__(self, k, qk, N, label_space, class_wise_seeds, mode="support"):
        assert k+qk <= N
        self.N = N
        self.InstDict = dict.fromkeys(label_space)

        if mode == 'support':
            start_idx, end_idx = 0, k
        elif mode == 'query':
            start_idx, end_idx = k, k+qk
        else:
            raise ValueError("[EpisodeSampler] Not supported mode: "+mode)

        for c, seed in zip(label_space, class_wise_seeds):
            perm = randPermutation(N, seed)
            self.InstDict[c] = perm[start_idx:end_idx]     # 对于支持集和查询集，取序列排列的前k个和接着的qk个元素

        # if mode == 'support':
        #     # 为每一个类，根据其种子生成抽样样本的下标
        #     for cla,seed in zip(label_space, class_wise_seeds):
        #         rd.seed(seed)
        #         # 注入该类对应的种子，然后抽样训练集
        #         self.InstDict[cla] = set(rd.sample([i for i in range(N)], k))
        #
        # elif mode == 'query':
        #     for cla,seed in zip(label_space, class_wise_seeds):
        #         rd.seed(seed)
        #         # 注入与训练集同类的种子，保证训练集采样相同
        #         train_instances = set(rd.sample([i for i in range(N)], k))
        #         # 查询集与训练集不同
        #         test_instances = set([i for i in range(N)]).difference(train_instances)
        #
        #         rd.seed(magicSeed())
        #         self.InstDict[cla] = rd.sample(test_instances, qk)

    def __iter__(self):
        batch = []
        for c, instances in self.InstDict.items():
            for i in instances:
                batch.append(self.N * c + i)
        return iter(batch)

    def __len__(self):          # 由于batch size将会设定为一次性取出所有sampler中的数据，因此只能sample一次
        return 1