import os
from tqdm import tqdm

from utils.file import loadJson, dumpIterable, dumpJson

def statValidJsonReport(dir_path, len_thresh=10,
                        class_dir=False,
                        name_prefix=None,
                        dump_valid_path=None):

    valid = invalid = too_short = total = 0
    valid_list = []

    for folder in os.listdir(dir_path):
        folder_path = dir_path+folder+'/'
        if class_dir:
            items = os.listdir(folder_path)
        else:
            items = [name_prefix+'.json']

        for item in items:
            total_length = 0
            total += 1
            print('#%d'%total, folder_path+item, end=': ')

            try:
                report = loadJson(folder_path+item)
                raw_file_name = report['target']['file']['name']
                for process in report['behavior']['processes']:
                    total_length += len(process['calls'])

                if total_length < len_thresh:
                    too_short += 1
                    print('too short:', total_length)
                else:
                    valid += 1
                    valid_list.append({'file':raw_file_name,
                                       'len':total_length,
                                       'rawPath':folder_path+item})
                    print('valid')
            except Exception as e:
                invalid += 1
                print('Error: ', str(e))

    print('Total:', total)
    print('Valid:', valid)
    print('Invalid:', invalid)
    print('Too Short:', too_short)

    if dump_valid_path is not None:
        dumpIterable(valid_list, title='valid_file_name', path=dump_valid_path)

if __name__ == '__main__':
    statValidJsonReport(dir_path='F:/result2/cuckoo/analyses/',
                        class_dir=False,
                        name_prefix='reports/report',
                        dump_valid_path='D:/datasets/LargePE-API-raw/reports/valid_file_list.json')