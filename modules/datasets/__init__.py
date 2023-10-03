from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .dukemtmcs import DukeMTMCs
from .market1501 import Market1501
from .market1501s import Market1501s
from .msmt17 import MSMT17
from .merge import MergedData
from .aicity import AIcityT1

from .synimgs import SyntheImgs

__factory = {
    'market1501': Market1501,
    'market1501s': Market1501s,
    'dukemtmc': DukeMTMC,
    'dukemtmcs': DukeMTMCs,
    'msmt17': MSMT17,
    'merged' : MergedData,
    'aicity' : AIcityT1,

    'synimgs': SyntheImgs,

    '': None
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'. 
        'market1501|dukemtmc|msmt' for merge datasets
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    
    #check merge datasets
    if "|" in name:
        list = name.split("|")
        return __factory['merged'](list, *args, **kwargs)
    
    #else
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, for_merge=False, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
