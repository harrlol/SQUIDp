import os
import os.path as osp
import pandas as pd
from typing import List

def auto_expand(path):
    """
    Detects and expands the path if it contains '~'.
    """
    if '~' in path:
        return osp.expanduser(path)
    return path

def get_ids(meta_df,  tissues: List[str], 
            species: str = 'Homo sapiens', st_technology: str = 'Xenium',):
    """
    Returns the sample ids according to the filter specified
    """
    meta_df = meta_df[meta_df['species'] == species]
    meta_df = meta_df[meta_df['st_technology'] == st_technology]
    meta_df = meta_df[meta_df['tissue'].isin(tissues)]
    return meta_df['id'].values