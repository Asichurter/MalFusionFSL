import os
import torch as t
import numpy as np
import PIL.Image as Image
import torchvision.transforms as T
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from utils.file import loadJson, dumpJson
from utils.manager import PathManager

##############################################################
# 将长序列截断，同时利用将原序列映射为词嵌入的下标
##############################################################
def _truncateAndMapTokenToIdx(token_seqs, mapping, max_len):
    for i,seq in enumerate(token_seqs):
        res_seq = []
        for token in seq[:max_len]:
            res_seq.append(mapping[str(token)])
        token_seqs[i] = t.LongTensor(res_seq)

    return token_seqs


##############################################################
# 将长序列截断，同时利用将原序列映射为词嵌入的下标
##############################################################
def _calculateImgMeanAndStd(img_path):
    data = []
    transformer = T.ToTensor()
    for folder in os.listdir(img_path):
        folder_path = img_path + folder + '/'
        for item in os.listdir(folder_path):
            image = Image.open(folder_path + item)
            image = transformer(image).squeeze()
            size = image.size
            data.append(image.sum()/(size(0)*size(1)))

    return np.mean(data),np.std(data)

##########################################################
# 本函数主要用于数据集的文件生成。
# 用于根据已经按类分好的JSON形式数据集，根据已经生成的嵌入矩阵和
# 词语转下标表来将数据集整合，token替换为对应的词语下标序列，同时pad，最后
# 将序列长度文件和数据文件进行存储的总调用函数。运行时要检查每个类的样本
# 数，也会按照最大序列长度进行截断。图像数据同样会被整合到文件中
##########################################################
def _packDataFile(api_path,
                  img_path,
                  w2idx_path,
                  seq_length_save_path,
                  api_data_save_path,
                  img_data_save_path,
                  num_per_class,
                  idx2cls_mapping_save_path=None,
                  max_seq_len=600,
                  img_suffix='jpg',
                  img_size=(1,256,256)):

    api_data_list = []
    img_data_list = []
    folder_name_mapping = {}

    print('Loading config data...')
    word2index = loadJson(w2idx_path)

    print('Calculating image statistics...')
    mean, std = _calculateImgMeanAndStd(img_path)
    transform = T.Compose([T.ToTensor(), T.Normalize([mean], [std])])

    print('Read main data...')
    for cls_idx, cls_dir in tqdm(enumerate(os.listdir(api_path))):
        api_folder_path = api_path + cls_dir + '/'
        img_folder_path = img_path + cls_dir + '/'

        assert num_per_class == len(os.listdir(api_folder_path)), \
            '数据集中类%s的样本数量%d与期望的样本数量不一致！'%\
            (cls_dir, len(os.listdir(api_folder_path)), num_per_class)

        for item in os.listdir(api_folder_path):
            report = loadJson(api_folder_path + item)
            apis = report['apis']
            api_data_list.append(apis)          # 添加API序列

            img_item = item[:-4]+img_suffix

            img = transform(Image.open(img_folder_path+img_item))
            # img_data_list = t.cat((img_data_list, t.unsqueeze(img, dim=0)), dim=0)
            img_data_list.append(img.tolist())

        folder_name_mapping[cls_idx] = cls_dir

    img_data_list = img_data_list[0:]       # 丢掉初始化时填入的0占位

    print('Converting...')
    api_data_list = _truncateAndMapTokenToIdx(api_data_list,
                                              word2index,
                                              max_seq_len)  # 转化为嵌入后的数值序列列表

    seq_length_list = {i:len(seq) for i,seq in enumerate(api_data_list)}   # 数据的序列长度

    api_data_list = pad_sequence(api_data_list, batch_first=True, padding_value=0)  # 数据填充0组建batch

    # 由于pad函数是根据输入序列的最大长度进行pad,如果所有序列小于最大长度,则有可能出现长度
    #　不一致的错误
    if api_data_list.size(1) < max_seq_len:
        padding_size = max_seq_len - api_data_list.size(1)

        zero_paddings = t.zeros((api_data_list.size(0),padding_size))
        api_data_list = t.cat((api_data_list, zero_paddings),dim=1)

    print('Dumping...')
    dumpJson(seq_length_list, seq_length_save_path)     # 存储序列长度到JSON文件
    if idx2cls_mapping_save_path is not None:
        dumpJson(folder_name_mapping, idx2cls_mapping_save_path)

    t.save(t.LongTensor(api_data_list), api_data_save_path)                   # 存储填充后的数据文件
    t.save(t.FloatTensor(img_data_list), img_data_save_path)

    print('Done')


##########################################################
# 同时将一个数据集中的train，validate和test同时打包成为文件格式
# 以便于读取到内存中
##########################################################
def packAllSubsets(dataset,
                   num_per_class,
                   max_seq_len=300):

    for subset in ['train', 'validate', 'test']:
        print("Packing", subset, "...")
        pm = PathManager(dataset, subset)
        _packDataFile(api_path=pm.apiDataFolder(),
                      img_path=pm.imgDataFolder(),
                      w2idx_path=pm.wordIndex(),
                      seq_length_save_path=pm.apiSeqLen(),
                      num_per_class=num_per_class,
                      api_data_save_path=pm.apiData(),
                      img_data_save_path=pm.imgData(),
                      idx2cls_mapping_save_path=pm.subsdetIdxClassMapper(),
                      max_seq_len=max_seq_len)