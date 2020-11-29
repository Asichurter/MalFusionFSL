import os

from utils.general import datasetTraverse
from utils.file import dumpJson, loadJson

def statNGramFrequency(dir_path,
                       N,
                       class_dir=True,
                       log_dump_path=None):

    def statNGramFrequencyInner(count_, filep_, report_, list_, dict_, **kwargs):
        apis = report_['apis']
        print('# %d' % count_, filep_, 'len=%d' % len(apis), end=' ')

        for i in range(len(apis)):
            if i+N >= len(apis):
                break
            ngram = '/'.join(apis[i:i+N])
            if ngram not in dict_:
                dict_[ngram] = 1
            else:
                dict_[ngram] += 1

            if len(list_) == 0:
                list_.append(1)
            else:
                list_[0] += 1

        return list_, dict_

    def statNGramFrequencyFNcb(reporter_, list_, dict_):
        for k in dict_:
            dict_[k] = dict_[k] / list_[0]
        if log_dump_path is not None:
            dumpJson(dict_, log_dump_path)

    datasetTraverse(dir_path=dir_path,
                    exec_kernel=statNGramFrequencyInner,
                    class_dir=class_dir,
                    final_callback=statNGramFrequencyFNcb)


def mapAndExtractTopKNgram(dir_path,
                           ngram_stat_log_path,
                           K,
                           N,
                           class_dir=True,
                           map_dump_path=None):

    ngram_fre = loadJson(ngram_stat_log_path)
    sorted_ngrams = sorted(ngram_fre.items(), key=lambda x:x[1], reverse=True)[:K]
    topk_ngrams = {x[0]:i+1 for i,x in enumerate(sorted_ngrams)}      # 将NGram映射为下标序号
    topk_ngrams['<PAD>'] = 0        # 0为pad

    def mapAndExtractTopKNgramInner(count_, filep_, report_, list_, dict_, **kwargs):
        print('# %d' % count_, end=' ')
        new_seq = []
        apis = report_['apis']

        for i in range(len(apis)):
            if i+N >= len(apis):
                break
            ngram = '/'.join(apis[i:i+N])
            if ngram in topk_ngrams:
                new_seq.append(topk_ngrams[ngram])

        new_report = {k:v for k,v in report_.items()}
        new_report['apis'] = new_seq

        dumpJson(new_report, filep_)
        return list_, dict_

    datasetTraverse(dir_path=dir_path,
                    exec_kernel=mapAndExtractTopKNgramInner,
                    class_dir=class_dir)

    if map_dump_path is not None:
        dumpJson(topk_ngrams, map_dump_path)
