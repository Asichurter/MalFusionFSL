import torch as t
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from utils.file import loadJson

class FusionDataset(Dataset):

    def __init__(self, api_data_path, img_data_path, seq_len_path, N):
        self.ApiData = t.load(api_data_path)
        self.ImgData = t.load(img_data_path)
        self.ApiSeqLen = [0] * len(self.ApiData)

        seq_lens = loadJson(seq_len_path)
        for i,l in seq_lens.items():
            self.ApiSeqLen[int(i)] = l

        assert len(self.ApiSeqLen) == len(self.ImgData) == len(self.ApiSeqLen)
        self.Datalen = len(self.ApiSeqLen)

        assert len(self.ApiData) % N == 0, "数据集长度不是N的整倍数: len=%d, N=%d"%()
        class_num = len(self.ApiData) // N
        self.TotalClassNum = class_num

        self.Labels = []
        for l in range(class_num):
            self.Labels += [l] * N

    def __getitem__(self, item):
        return self.ApiData[item], self.ImgData[item], self.ApiSeqLen[item], self.Labels[item]      # TODO: 封装为元组隔离数据与标签？

    def __len__(self):
        return self.Datalen


class SeqDataset(Dataset):

    def __init__(self, api_data_path, seq_len_path, N):
        self.ApiData = t.load(api_data_path)
        self.ApiSeqLen = [0] * len(self.ApiData)

        seq_lens = loadJson(seq_len_path)
        for i,l in seq_lens.items():
            self.ApiSeqLen[int(i)] = l

        assert len(self.ApiSeqLen) == len(self.ApiSeqLen)
        self.Datalen = len(self.ApiSeqLen)

        assert len(self.ApiData) % N == 0, "[SeqDataset] 数据集长度不是N的整倍数: len=%d, N=%d"%()
        class_num = len(self.ApiData) // N
        self.TotalClassNum = class_num

        self.Labels = []
        for l in range(class_num):
            self.Labels += [l] * N

    def __getitem__(self, item):
        return self.ApiData[item], self.ApiSeqLen[item], self.Labels[item]

    def __len__(self):
        return self.Datalen

    def getCollectFn(self):

        def seqCollectFn(data):
            seqs, lens, labels = [], [], []
            for seq,len_,label in data:
                seqs.append(seq.tolist())
                lens.append(len_)
                labels.append(label)
            # 没有按照序列长度排序，需要在pack时设置enforce_sort = False
            return t.LongTensor(seqs), lens, t.LongTensor(labels)

        return seqCollectFn


class ImgDataset(Dataset):

    def __init__(self, img_data_path, N):
        self.ImgData = t.load(img_data_path)

        self.Datalen = len(self.ImgData)

        assert len(self.ImgData) % N == 0, "[ImgDataset] 数据集长度不是N的整倍数: len=%d, N=%d"%()
        class_num = len(self.ImgData) // N
        self.TotalClassNum = class_num

        self.Labels = []
        for l in range(class_num):
            self.Labels += [l] * N

    def __getitem__(self, item):
        return self.ImgData[item], self.Labels[item]

    def __len__(self):
        return self.Datalen

    def getCollectFn(self):

        def imgCollectFn(data):
            imgs, labels = [], []
            for img, label in data:
                imgs.append(img.tolist())
                labels.append(label)
            return t.Tensor(imgs), t.LongTensor(labels)

        return imgCollectFn

class FusedDataset:

    def __init__(self, api_data_path, img_data_path, seq_len_path, N, data_source):
        none_flag = True
        if "sequence" in data_source:
            self.SeqDataset = SeqDataset(api_data_path, seq_len_path, N)
            self.TotalClassNum = self.SeqDataset.TotalClassNum
            none_flag = False
        else:
            self.SeqDataset = None
        if "image" in data_source:
            self.ImgDataset = ImgDataset(img_data_path, N)
            self.TotalClassNum = self.ImgDataset.TotalClassNum
            none_flag = False
        else:
            self.ImgDataset = None
        assert not none_flag, "[FusedDataset] 必须指定sequence和image其中至少一种数据源"

    def sample(self, sampler, batch_size):
        ret = [None, None, None, None]    # seqs,imgs,lens,labels

        if self.SeqDataset is not None:
            seq_loader = DataLoader(self.SeqDataset, batch_size=batch_size,
                                    sampler=sampler, collate_fn=self.SeqDataset.getCollectFn())
            seqs, lens, labels = seq_loader.__iter__().__next__()
            ret[0], ret[2], ret[3] = seqs, lens, labels

        if self.ImgDataset is not None:
            img_loader = DataLoader(self.ImgDataset, batch_size=batch_size,
                                    sampler=sampler, collate_fn=self.ImgDataset.getCollectFn())
            imgs, labels = img_loader.__iter__().__next__()
            ret[1], ret[3] = imgs, labels

        return ret