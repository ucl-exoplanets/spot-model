from numbers import Number
from typing import Union, Iterable, Optional

import numpy as np

NumericOrIterable = Optional[Union[Number, Iterable[Number]]]

def parse_args_lists(*args, same_length: bool=True):
    out = []    
    for arg in args:    
        if not hasattr(arg, '__len__'):
            out.append([arg])
        else:
            out.append(list(arg))
    if same_length:
        assert (np.diff([len(arg) for arg in out])==0).all()
    return out

def spher_to_cart(lat: NumericOrIterable, lon: NumericOrIterable):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z
