import os
import platform


def joinPath(*ps, is_dir=False):
    ret = ""
    for p in ps[:-1]:
        if not p.endswith('/'):
            p += "/"
        ret += p
    if is_dir and not ps[-1].endswith('/'):
        return ret + ps[-1] + '/'
    else:
        return ret + ps[-1]


def rmAll(path):
    if path[-1] == '/':
        path = path[:-1]
    system = platform.system()
    if system == 'Linux':
        assert os.system(f"rm -rf {path}/*") == 0, "Fail to run rmAll"
    elif system == 'Windows':
        win_style_path = path.replace('/', '\\')
        # windows平台，先删除文件夹内所有内容，再重新新建一个同名空文件夹
        assert os.system(rf"rmdir /s/q {win_style_path}") == 0, "Fail to run rmAll"
        os.mkdir(win_style_path)
    else:
        raise NotImplementedError(f"Not supported system for rmAll: {system}")
