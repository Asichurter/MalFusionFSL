from preprocessing.normalization import removeAPIRedundancy
from preprocessing.ngram import statNGramFrequency, mapAndExtractTopKNgram

# 调用顺序：rmRedundancy -> statNGramFre ->

# removeAPIRedundancy(dir_path='/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/',
#                         class_dir=True)

# statNGramFrequency(dir_path='/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/',
#                    N=3,
#                    class_dir=True,
#                    log_dump_path='/home/omnisky/NewAsichurter/FusionData/reports/Per40_3gram_stat.json')

mapAndExtractTopKNgram(dir_path='/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/',
                       ngram_stat_log_path='/home/omnisky/NewAsichurter/FusionData/reports/Per40_3gram_stat.json',
                       K=5000,
                       N=3,
                       class_dir=True,
                       map_dump_path='/home/omnisky/NewAsichurter/FusionData/reports/Per40_3gram_map.json')

