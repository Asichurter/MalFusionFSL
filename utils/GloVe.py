# -*- coding: utf-8

from glove import Glove, Corpus
import numpy as np
import json
import os
from tqdm import tqdm
import argparse

def dumpJson(obj, path, indent=4):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)

def loadJson(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return j

############################################
# 本函数用于从已经处理好的json文件中收集所有样本的api
# 序列用于无监督训练嵌入。返回的是序列的列表。
############################################
def aggregateApiSequences(path, is_class_dir=True):

    seqs = []

    for folder in tqdm(os.listdir(path)):
        folder_path = path + folder + '/'

        if is_class_dir:            # 如果是类文件夹，则整个路径下都是需要检索的JSON
            items = os.listdir(folder_path)
        else:                       # 如果是个体文件夹，路径下只有 文件夹名+.JSON 才是需要检索的
            items = [folder + '.json']

        for item in items:
            try:
                report = loadJson(folder_path + item)
                apis = report['apis']

                if len(apis) > 0:
                    seqs.append(apis)
            except Exception as e:
                print(folder, item, e)
                exit(-1)

    return seqs

def getGloveEmbedding(seqs, size=300, window=10, epochs=20):
    corpus = Corpus()
    corpus.fit(seqs, window=window)

    glove = Glove(no_components=size, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=epochs,
              verbose=True)

    return corpus.dictionary, glove.word_vectors

###############################################
# 使用GloVe训练词向量初始值。
# 注意： 本函数只能在Python2.7环境下使用！
###############################################
def trainGloVe(seqs,
               size=300,
               save_matrix_path=None,
               save_word2index_path=None,
               padding=True,
               **kwargs):

    print('Traning GloVe...')
    word2index, matrix = getGloveEmbedding(seqs, size, **kwargs)

    if padding:
        pad_matrix = np.zeros((1, matrix.shape[1]))
        matrix = np.concatenate((pad_matrix, matrix), axis=0)

        for k in word2index.keys():
            word2index[k] = word2index[k] + 1 if padding else word2index[k]  # 由于idx=0要留给padding，因此所有的下标都加1
        word2index['<PAD>'] = 0

    print('Saving...')
    if save_matrix_path:
        np.save(save_matrix_path, matrix)

    if save_word2index_path:
        dumpJson(word2index, save_word2index_path)

    if save_matrix_path is None and save_word2index_path is None:
        return matrix, word2index

    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--basepath', type=str)
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-s', '--size', type=int)
    parser.add_argument('-t', '--type', type=str, default='all')

    args = parser.parse_args()

    seqs = aggregateApiSequences(args.basepath + args.dataset + "/%s/api/"%args.type)   # 数据集中不仅包含api数据，还有pe数据
    trainGloVe(seqs,
               size=args.size,
               save_matrix_path=args.basepath + args.dataset + "/data/matrix.npy",      # 2.7环境没有torch，直接存为npy格式
               save_word2index_path=args.basepath + args.dataset + "/data/wordMap.json")