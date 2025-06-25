import os
import os.path as osp

def auto_expand(path):
    """
    Detects and expands the path if it contains '~'.
    """
    if '~' in path:
        return osp.expanduser(path)
    return path