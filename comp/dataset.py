import torch as t
from torch.utils.data.dataset import Dataset

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