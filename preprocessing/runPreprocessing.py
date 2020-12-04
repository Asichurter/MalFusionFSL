from preprocessing.normalization import removeAPIRedundancy
from preprocessing.ngram import statNGramFrequency, mapAndExtractTopKNgram
from preprocessing.wdembedding import trainGloVe
from preprocessing.dataset.structure import makeDatasetDirStruct
from preprocessing.dataset.rename import renamePEbyMD5fromApi
from preprocessing.image import convertDir2Image, convert
from preprocessing.dataset.split import splitDataset
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

# makeDatasetDirStruct(base_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/')

# trainGloVe(base_path='/home/omnisky/NewAsichurter/FusionData/datasets/',
#            dataset='LargePE-Per40',
#            size=300,
#            type='all')

# renamePEbyMD5fromApi(api_dir_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/all/api/',
#                      pe_dir_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/all/pe/')

# convertDir2Image(dir_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/PEs/',
#                  dst_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/all/img/')

# splitDataset(dataset_path='/home/omnisky/NewAsichurter/FusionData/datasets/LargePE-Per40/',
#              validate_ratio=30,
#              test_ratio=30)

packAllSubsets("LargePE-Per40", num_per_class=40, max_seq_len=300)




