"""Utility functions and type

Types:

    NumOrIt = Optional[Union[Number, Iterable[Number]]]

Functions:

    parse_args_lists(*args, same_length: bool = True) -> Iterable:
    spher_to_cart(lat: NumOrIt, lon: NumOrIt) -> Tuple[NumOrIt, NumOrIt, NumOrIt]:
    
"""
from numbers import Number
from typing import Union, Iterable, Optional, Tuple

import numpy as np

NumOrIt = Optional[Union[Number, Iterable[Number]]]


def parse_args_lists(*args, same_length: bool = True) -> Iterable:
    """Parse a list of arguments into a list of lists.

    Args:
        same_length (bool, optional): whether to require arguments to be of same length. 
            Defaults to True.

    Returns:
        Iterable: list of arguments parsed into lists
    """
    out = []
    for arg in args:
        if not hasattr(arg, '__len__'):
            out.append([arg])
        else:
            out.append(list(arg))
    if same_length:
        assert (np.diff([len(arg) for arg in out]) == 0).all()
    return out


def spher_to_cart(lat: NumOrIt, lon: NumOrIt) -> Tuple[NumOrIt, NumOrIt, NumOrIt]:
    """Convert lat/lon to spherical cartesian coordinates.

    Args:
        lat (NumOrIt): latitude in degrees 
        lon (NumOrIt): longitude in degrees

    Returns:
        Tuple[NumOrIt, NumOrIt, NumOrIt]: cartesian coordinates x, y, z
    """
    if not (np.greater_equal(lat, -90).all() and np.less_equal(lat, 90).all()):
        raise ValueError('latitude is defined between -90 and 90°')
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z
