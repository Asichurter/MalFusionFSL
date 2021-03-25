import os
import shutil
from tqdm import tqdm

from utils.file import loadJson
from utils.log import Reporter


def collectPEwithAPI(api_dir_path,
                     pe_dir_path,
                     dst_path,
                     class_dir=True,
                     name_prefix=None,      # 后缀默认是json，因为要读取json文件
                     log_dump_path=None):

    print("[CollectPEwithAPI] Preparing...")
    pe_folder_map = {folder: os.listdir(pe_dir_path+folder) for folder in os.listdir(pe_dir_path)}
    reporter = Reporter()

    print("[CollectPEwithAPI] Starting...")
    for folder in tqdm(os.listdir(api_dir_path)):
        folder_path = api_dir_path+folder+'/'
        if class_dir:
            items = os.listdir(folder_path)
            os.mkdir(dst_path+folder+'/')
            dst_folder = dst_path+folder+'/'
        else:
            items = [name_prefix+'.json']       # 对于没有分类的文件，移动时不重新创建文件夹
            dst_folder = dst_path

        for item in items:
            try:
                report = loadJson(folder_path+item)
                name = report['name']

                found_flag = False
                for pe_folder in pe_folder_map:
                    if name in pe_folder_map[pe_folder]:
                        shutil.copy(pe_dir_path+pe_folder+'/'+name, dst_folder+name)
                        found_flag = True
                        break
                if not found_flag:
                    reporter.logError(entity=name,
                                      msg="File not found")
                else:
                    reporter.logSuccess()
            except Exception as e:
                reporter.logError(entity=folder_path+item,
                                  msg=str(e))

    reporter.report()
    if log_dump_path is not None:
        reporter.dump(log_dump_path)

if __name__ == '__main__':
    collectPEwithAPI(api_dir_path='E:/FSL数据/项目2-动态分析数据/JSONs/virushare-20-3gram-tfidf/all/',
                     pe_dir_path='E:/FSL数据/项目1-静态分析数据/peimages/PEs/virushare_20/all/',
                     dst_path='F:/FSL_mal_data/datasets/virushare-20/all/pe/',
                     class_dir=True,
                     log_dump_path=None)


