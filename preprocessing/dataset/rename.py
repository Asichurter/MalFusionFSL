import os

from utils.general import datasetTraverse

def renamePEbyMD5fromApi(api_dir_path, pe_dir_path):

    def renamePEbyMD5fromApiInner(count_, filep_, report_, list_, dict_, **kwargs):
        print("# %d"%count_, filep_, end=' ')

        md5 = report_['md5']
        folder = '.'.join(report_['name'].split('.')[:3])
        item = report_['name']

        os.rename(pe_dir_path+folder+'/'+item,
                  pe_dir_path+folder+'/'+md5)

        return list_, dict_

    datasetTraverse(dir_path=api_dir_path,
                    exec_kernel=renamePEbyMD5fromApiInner,
                    class_dir=True)
