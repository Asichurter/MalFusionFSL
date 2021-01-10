import os

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
    assert os.system(f"rm -rf {path}/*") == 0, "Fail to run rmAll"
