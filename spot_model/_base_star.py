
import warnings

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap


class _BaseStar(object):
    """Palermo star model"""
    star_value = 0
    spot_value = 1
    planet_value = 2
    
    def __init__(self, nr: int = 1000,  nth: int = 1000):
        self.nr = nr
        self.nth = nth
        self.radii = (0.5 + np.arange(self.nr)) / self.nr
        self.mu = np.sqrt(1. - self.radii**2)

        self.deltar = 1. / self.nr
        self.deltath = 2. * np.pi / self.nth

        theta_edges = np.linspace(0., 2. * np.pi, nth+1)
        self.theta = (theta_edges[1:] + theta_edges[:-1]) / 2.

        self.RR, self.TT = np.meshgrid(self.radii, self.theta, copy=False)
        self.Z = self.RR * np.sin(self.TT)
        self.Y = self.RR * np.cos(self.TT)
        self.Th = np.arccos(self.Z)
        self.Ph = np.arcsin(self.Y / np.sin(self.Th))
        self.X = np.sin(self.Th) * np.cos(self.Ph)

    def show(self, spotted_mask,  yp=None, zp=None, rp=None, ax=None, axis=False,
             cm=LinearSegmentedColormap.from_list("custom", ["#fff305", "#ffa805"])):
        if ax is None:
            _, ax = plt.subplots()
        plt.pcolormesh(self.Y, self.Z, spotted_mask,
                       shading='nearest', cmap=cm, antialiased=True)  # 'YlGn', vmin=0, vmax=3)
        plt.axhline(0, color='black', label='equator',
                    linestyle='dashed', linewidth=0.5)
        plt.axvline(0, color='black', label='meridian',
                    linestyle='dashed', linewidth=0.5)
        ax.add_patch(plt.Circle((0, 0), 1, edgecolor='black',
                     facecolor='none', linewidth=0.4))

        # Show planet path if yp is provided
        if yp is not None:
            if hasattr(rp, '__len__'):
                warnings.warn(
                    'only one radius is supported, taking the largest radius')
                rp = np.max(rp)
            if not hasattr(yp, '__len__'):
                yp = [yp]
                assert not hasattr(zp, '__len__')
                zp = [zp]
                list_alpha = [1.]
            else:
                assert len(yp) == len(zp)
                list_alpha = np.linspace(0.5, 1, len(yp))
            for i in range(len(zp)):
                ax.add_patch(plt.Circle((yp[i], zp[i]), rp, edgecolor=None,
                                        facecolor='black', alpha=list_alpha[i]))
        plt.axis('equal')
        if not axis:
            plt.axis('off')
        else:
            plt.xlabel('Y')
            plt.ylabel('Z', rotation=0)
        return ax
