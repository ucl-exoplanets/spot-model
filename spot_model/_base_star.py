import warnings
from numbers import Number
from typing import Union, Iterable, Optional

import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

NumericOrIterable = Optional[Union[Number, Iterable[Number]]]

default_colormap = LinearSegmentedColormap.from_list(
    "custom", ["#fff305", "#ffa805"])


class _BaseStar(object):
    """Base class for star model as disk parameterised in polar coordinates."""
    star_value = 0
    spot_value = 1
    planet_value = 2

    def __init__(self, nr: int = 1000,  nth: Optional[int] = None, debug: bool = True):
        assert isinstance(nr, int) and nr > 0
        self.nr = nr
        self.radii = (0.5 + np.arange(self.nr)) / self.nr
        self.mu = np.sqrt(1. - self.radii**2)
        self.deltar = 1. / self.nr
        self.debug = debug

        if nth is not None:
            assert isinstance(nth, int)
            assert nth > 0
            self.nth = nth
            self.deltath = 2. * np.pi / self.nth
            theta_edges = np.linspace(0., 2. * np.pi, nth+1)
            self.theta = (theta_edges[1:] + theta_edges[:-1]) / 2.

            self.RR, self.TT = np.meshgrid(self.radii, self.theta, copy=False)
            self.Z = self.RR * np.sin(self.TT)
            self.Y = self.RR * np.cos(self.TT)
            self.Th = np.arccos(self.Z)
            self.Ph = np.arcsin(self.Y / np.sin(self.Th))
            self.X = np.sin(self.Th) * np.cos(self.Ph)
            self._mask = np.zeros([self.nth, self.nr])
        else:
            self._mask = None
            

    def show(self, yp: Number = None, zp: Number = None, rp: Number = None,
             ax: Optional[Axes] = None, projection: str = 'polar', show_axis: bool = False,
             cm: Colormap = default_colormap) -> Axes:
        """Display the star with spot(s) and optional transiting planet.

        Args:
            spotted_mask (ndarray): spotted star mask (2D ndarray)
            yp (Number, optional): planet y position. Defaults to None.
            zp (Number, optional): planet z position. Defaults to None.
            rp (NumericOrIterable, optional): planet radius - only one radius supported for now. Defaults to None.
            ax (Optional[Axes], optional): base matplotlib axe to use. Defaults to None.
            projection (str): matplotlib axe projection to use should a new axis be defined. Defaults to 'polar' 
            show_axis (bool, optional): whether to show the star prime meridian and equator. Defaults to False.
            cm (Colormap, optional): colormap for the star. Defaults to default_colormap.

        Returns:
            Axes: matplotlib Axe with the created plot
        """
        if self._mask is None:
            raise RuntimeError("show needs an accessible _mask attribute")
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=projection)

        if ax.name == 'polar':
            ax.pcolormesh(self.TT, self.RR, self._mask,
                          shading='nearest', cmap=cm, antialiased=True, vmin=0, vmax=1)
            ax.vlines([0, np.pi, np.pi/2, -np.pi/2], 0, 1.2,
                      color='black', linestyle='dashed', linewidth=0.5)
            # ax.hlines(1, -np.pi, np.pi)
            ax.add_patch(plt.Circle((0, 0), 1, edgecolor='black',
                                    facecolor='none', linewidth=0.4, transform=ax.transData._b))

        elif ax.name == 'rectilinear':
            ax.pcolormesh(self.Y, self.Z, self._mask,
                          shading='nearest', cmap=cm, antialiased=True, vmin=0, vmax=1)
            ax.hlines(0, -1.2, 1.2, color='black', label='equator',
                        linestyle='dashed', linewidth=0.5)
            ax.vlines(0, -1.2, 1.2, color='black', label='meridian',
                        linestyle='dashed', linewidth=0.5)

            ax.add_patch(plt.Circle((0, 0), 1, edgecolor='black',
                                    facecolor='none', linewidth=0.4))
            plt.axis('equal')

        else:
            raise NotImplementedError

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
                if ax.name == 'polar':
                    ax.add_patch(plt.Circle((yp[i], zp[i]), rp, edgecolor=None,
                                            facecolor='black', alpha=list_alpha[i], transform=ax.transData._b))
                elif ax.name == 'rectilinear':
                    ax.add_patch(plt.Circle((yp[i], zp[i]), rp, edgecolor=None,
                                            facecolor='black', alpha=list_alpha[i]))
                else:
                    raise NotImplementedError

        if not show_axis:
            plt.axis('off')
        else:
            plt.xlabel('Y')
            plt.ylabel('Z')
        return ax
