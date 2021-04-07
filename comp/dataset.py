import torch as t
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random

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


def seqCollectFn(data):
    seqs, lens, labels = [], [], []
    for seq, len_, label in data:
        seqs.append(seq)
        lens.append(len_)
        labels.append(label)
    # 没有按照序列长度排序，需要在pack时设置enforce_sort = False
    return t.cat(seqs, dim=0), lens, t.cat(labels, dim=0)


class SeqDataset(Dataset):

    def __init__(self, api_data_path, seq_len_path, N):
        self.ApiData = t.load(api_data_path).cuda().unsqueeze(1)    # 在第2维度上扩充，以便下标取时不用再unsqueeze
        self.ApiSeqLen = [0] * len(self.ApiData)

        seq_lens = loadJson(seq_len_path)
        for i,l in seq_lens.items():
            self.ApiSeqLen[int(i)] = l

        assert len(self.ApiSeqLen) == len(self.ApiSeqLen)
        self.Datalen = len(self.ApiSeqLen)

        assert len(self.ApiData) % N == 0, "[SeqDataset] 数据集长度不是N的整倍数: len=%d, N=%d"%()
        class_num = len(self.ApiData) // N
        self.TotalClassNum = class_num

        labels = []
        for l in range(class_num):
            labels += [l] * N
        self.Labels = t.LongTensor(labels).cuda().unsqueeze(1)  # 在第2维度上扩充，以便下标取时不用再unsqueeze

    def __getitem__(self, item):
        return self.ApiData[item], self.ApiSeqLen[item], self.Labels[item]

    def __len__(self):
        return self.Datalen

    def getCollectFn(self):
        return seqCollectFn

def imgCollectFn(data):
    imgs, labels = [], []
    for img, label in data:
        imgs.append(img)
        labels.append(label)
    return t.cat(imgs, dim=0), t.cat(labels, dim=0)


class ImgDataset(Dataset):

    def __init__(self, img_data_path, N,
                 crop_size=224,
                 rotate=True):
        self.ImgData = t.load(img_data_path).cuda().unsqueeze(1)    # 在第2维度上扩充，以便下标取时不用再unsqueeze
        self.CropSize = crop_size
        self.W = self.ImgData.size(-1)
        self.Rotate = rotate

        self.Datalen = len(self.ImgData)

        assert len(self.ImgData) % N == 0, "[ImgDataset] 数据集长度不是N的整倍数: len=%d, N=%d"%()
        class_num = len(self.ImgData) // N
        self.TotalClassNum = class_num

        labels = []
        for l in range(class_num):
            labels += [l] * N
        self.Labels = t.LongTensor(labels).cuda().unsqueeze(1)  # 在第2维度上扩充，以便下标取时不用再unsqueeze

    def _rotate_img(self, img):
        if not self.Rotate:
            return img

        rot_var = random.choice([0,1,2,3])
        return t.rot90(img, rot_var, dims=(2,3))

    def _crop_img(self, img):
        if self.CropSize is None:
            return img

        crop_size = self.CropSize
        w = self.W

        bound_width = w - crop_size
        x_rd, y_rd = random.randint(0, bound_width), random.randint(0, bound_width)
        img = img[:, :, x_rd:x_rd + crop_size, y_rd:y_rd + crop_size]           # 留两个维度出来，一个是通道维度，一个是扩张cat用的维度

        return img

    def __getitem__(self, item):
        img = self.ImgData[item]
        img = self._crop_img(img)
        img = self._rotate_img(img)
        return img, self.Labels[item]  # 留出一个维度以便cat

    def __len__(self):
        return self.Datalen

    def getCollectFn(self):
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

        self.SupportBatchSampler = None
        self.QueryBatchSampler = None

        self.SupportBatchSeqLoader = None
        self.SupportBatchImgLoader = None
        self.QueryBatchSeqLoader = None
        self.QueryBatchImgLoader = None



    def addBatchSampler(self, support_batch_sampler, query_batch_sampler):
        self.SupportBatchSampler = support_batch_sampler
        self.QueryBatchSampler = query_batch_sampler

        if self.SeqDataset is not None:
            self.SupportBatchSeqLoader = DataLoader(self.SeqDataset,
                                                    batch_sampler=self.SupportBatchSampler,
                                                    num_workers=0,
                                                    collate_fn=self.SeqDataset.getCollectFn()).__iter__()
            self.QueryBatchSeqLoader = DataLoader(self.SeqDataset,
                                                  batch_sampler=self.QueryBatchSampler,
                                                  num_workers=0,
                                                  collate_fn=self.SeqDataset.getCollectFn()).__iter__()

        if self.ImgDataset is not None:
            self.SupportBatchImgLoader = DataLoader(self.ImgDataset,
                                                    batch_sampler=self.SupportBatchSampler,
                                                    num_workers=0,
                                                    collate_fn=self.ImgDataset.getCollectFn()).__iter__()
            self.QueryBatchImgLoader = DataLoader(self.ImgDataset,
                                                  batch_sampler=self.QueryBatchSampler,
                                                  num_workers=0,
                                                  collate_fn=self.ImgDataset.getCollectFn()).__iter__()

    def _sampleSupportSeqByBatch(self):
        return self.SupportBatchSeqLoader.__next__()

    def _sampleSupportImgByBatch(self):
        return self.SupportBatchImgLoader.__next__()

    def _sampleQuerySeqByBatch(self):
        return self.QueryBatchSeqLoader.__next__()

    def _sampleQueryImgByBatch(self):
        return self.QueryBatchImgLoader.__next__()

    def sampleByBatch(self, mode='support'):
        ret = [None, None, None, None]  # seqs,imgs,lens,labels

        if mode == 'support':
            seq_sample_func = self._sampleSupportSeqByBatch
            img_sampler_func = self._sampleSupportImgByBatch
        else:
            seq_sample_func = self._sampleQuerySeqByBatch
            img_sampler_func = self._sampleQueryImgByBatch

        if self.SeqDataset is not None:
            seqs, lens, labels = seq_sample_func()
            ret[0], ret[2], ret[3] = seqs, lens, labels

        if self.ImgDataset is not None:
            imgs, labels = img_sampler_func()
            ret[1], ret[3] = imgs, labels

        return ret

    def sample(self, sampler, batch_size):
        ret = [None, None, None, None]    # seqs,imgs,lens,labels

        if self.SeqDataset is not None:
            seq_loader = DataLoader(self.SeqDataset, batch_size=batch_size,
                                    sampler=sampler, collate_fn=self.SeqDataset.getCollectFn(),
                                    num_workers=0)
            seqs, lens, labels = seq_loader.__iter__().__next__()
            ret[0], ret[2], ret[3] = seqs, lens, labels

        if self.ImgDataset is not None:
            img_loader = DataLoader(self.ImgDataset, batch_size=batch_size,
                                    sampler=sampler, collate_fn=self.ImgDataset.getCollectFn(),
                                    num_workers=0)
            imgs, labels = img_loader.__iter__().__next__()
            ret[1], ret[3] = imgs, labels

        return ret

