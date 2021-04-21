from preprocessing.dataset.collect import collectPEwithAPI
from preprocessing.dataset.sample import sampleClassWiseData
from preprocessing.normalization import removeAPIRedundancy
from preprocessing.ngram import statNGramFrequency, mapAndExtractTopKNgram
from preprocessing.word_embedding import trainGloVe
from preprocessing.dataset.structure import makeDatasetDirStruct
from preprocessing.dataset.rename import renamePEbyMD5fromApi
from preprocessing.image import convertDir2Image, convert
from preprocessing.dataset.split import *
from preprocessing.pack import packAllSubsets
from utils.manager import PathManager

# 调用顺序：rmRedundancy -> statNGramFre -> mapAndExtract

# removeAPIRedundancy(dir_path='/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/',
#                         class_dir=True)
#
# statNGramFrequency(dir_path='/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/',
#                    N=3,
#                    class_dir=True,
#                    log_dump_path='/home/omnisky/NewAsichurter/FusionData/reports/Per40_3gram_stat.json')
#
# mapAndExtractTopKNgram(dir_path='/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/',
#                        ngram_stat_log_path='/home/omnisky/NewAsichurter/FusionData/reports/Per40_3gram_stat.json',
#                        K=5000,
#                        N=3,
#                        class_dir=True,
#                        map_dump_path='/home/omnisky/NewAsichurter/FusionData/reports/Per40_3gram_map.json')

# makeDatasetDirStruct(base_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per35/')
#
# sampleClassWiseData(dst_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per35/all/api/',
#                     log_file_path="/home/omnisky/NewAsichurter/FusionData/reports/class_stat_after_processing_log.json",
#                     num_per_class=35)
#
pm = PathManager(dataset="virushare-20")
# trainGloVe(base_path=pm.rootBase(),
#            dataset='virushare-20',
#            size=300,
#            type='all')

# renamePEbyMD5fromApi(api_dir_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/all/api/',
#                      pe_dir_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/all/pe/')

# convertDir2Image(dir_path='F:/FSL_mal_data/datasets/virushare-20/all/pe/',
#                  dst_path='F:/FSL_mal_data/datasets/virushare-20/all/img/')

# splitDataset(dataset_path=pm.datasetBase(),
#              validate_ratio=20,
#              test_ratio=20)

revertDatasetSplit(dataset="virushare-20", dump_path=pm.dataBase()+"split_1.json")
# dumpDatasetSplitStruct(pm.datasetBase(),
#                        pm.dataBase()+'split_1.json',
#                        desc=['common split'])
packAllSubsets("virushare-20", num_per_class=20, max_seq_len=300)






