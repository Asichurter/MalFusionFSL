import os
from tqdm import tqdm

from utils.file import loadJson, dumpJson

processed_dataset_path = "/home/omnisky/NewAsichurter/FusionData/api/LargePE-API-Per40/"
stairs = [20,25,30,35,40,45]
length_thresh = 10
dump_stat_log_path = "/home/omnisky/NewAsichurter/FusionData/reports/class_stat_after_processing_log.json"

all_class_item_ok_dict = {}

for folder in tqdm(os.listdir(processed_dataset_path)):
    item_ok_list = []
    for json_item in os.listdir(processed_dataset_path+folder):
        item_path = processed_dataset_path+folder + '/' + json_item
        report = loadJson(item_path)
        leng = len(report['apis'])
        if leng >= length_thresh:
            item_ok_list.append(item_path)

    all_class_item_ok_dict[folder] = item_ok_list


print("Final Statistics")
for stair in stairs:
    count = 0
    for k,l in all_class_item_ok_dict.items():
        if len(l) >= stair:
            count += 1
    print("More than %d items: %d"%(stair, count))

dumpJson(all_class_item_ok_dict, dump_stat_log_path)
