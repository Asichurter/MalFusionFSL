
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
