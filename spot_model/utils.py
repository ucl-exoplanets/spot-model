import numpy as np

def spher_to_cart(lat, lon):
    lat = lat * np.pi / 180
    lon = lon * np.pi / 180
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z
