
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap


class BaseStarModel(object):
    """Palermo star model"""
    star_value = 0
    spot_value = 1
    planet_value = 2

    def __init__(self, nr: int = 100,  nth: int = 100):
        self.nr = nr
        self.nth = nth
        self.radii = (0.5 + np.arange(self.nr)) / self.nr
        self.mu = np.sqrt(1. - self.radii**2)

        self.deltar = 1. / self.nr
        self.deltath = 2. * np.pi / self.nth

        theta_edges = np.linspace(0., 2. * np.pi, nth+1)
        self.theta = (theta_edges[1:] + theta_edges[:-1]) / 2.

        RR, TT = np.meshgrid(self.radii, self.theta, copy=False)
        self.Z = RR*np.sin(TT)
        self.Y = RR*np.cos(TT)
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
                    linestyle='dashed', linewidth=0.7)
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


def spher_to_cart(lat, lon):
    x = np.sin(lat) * np.cos(lon)
    y = np.sin(lat) * np.sin(lon)
    z = np.cos(lat)
    return x, y, z


class StarModel(BaseStarModel):
    def create_mask_feat(self, y, z, rfeat, x=None):
        r0 = np.sqrt(y**2 + z**2)
        r_min = r0 - rfeat
        r_max = r0 + rfeat

        value = np.arctan2(z, y)
        theta0 = value if value >= 0 else 2. * np.pi + value
        d_theta = np.sqrt(2.) * rfeat / r0
        theta_min = theta0 - d_theta
        theta_max = theta0 + d_theta

        indr = np.where((self.radii >= r_min) & (self.radii <= r_max))[0]
        if r0 <= rfeat:
            indth = np.arange(self.nth, dtype=int)
        else:
            if theta_min >= 0.:
                indth = np.where((self.theta >= theta_min)
                                 & (self.theta <= theta_max))[0]
            else:
                theta_min += 2. * np.pi
                indth = np.append(np.where(self.theta <= theta_max)[0],
                                  np.where(self.theta >= theta_min)[0])
        dd = np.sqrt(((self.X[np.ix_(indth, indr)] - x)**2. if x else 0)+
                     (self.Y[np.ix_(indth, indr)] - y)**2. +
                     (self.Z[np.ix_(indth, indr)] - z)**2.)
        dth = 2. * np.arcsin(dd / 2.)
        dth[dth > np.pi / 2.] = 0
        dd *= np.cos(dth / 2.)
        return dd <= rfeat, indr, indth

    def create_mask_spot(self, lat, lon, rspot):
        x, y, z = spher_to_cart(lat, lon)
        mask, indr, indth = self.create_mask_feat(y, z, rspot, x=x)
        mask = mask.astype(int)
        if self.spot_value != 1:
            mask[mask] = self.spot_value
        return mask, indr, indth

    def create_mask_planet(self, y, z, rplanet):
        mask, indr, indth = self.create_mask_feat(y, z, rplanet)
        mask = mask.astype(int)
        mask[mask] = self.planet_value
        return mask, indr, indth

    def lc_mask(self, lat, lon, rspot):
        mask = np.zeros([self.nth, self.nr], bool)
        if isinstance(lat, (int, float)):
            lat = [lat]
        if isinstance(lon, (int, float)):
            lon = [lon]
        if isinstance(rspot, (int, float)):
            rspot = [rspot]
        for i in range(len(lat)):
            lat1 = (np.pi / 2. - lat[i] * np.pi / 180.)
            lon1 = lon[i] * np.pi / 180.
            mask1, indr, indtheta = self.create_mask_spot(lat1, lon1, rspot[i])
            mask[np.ix_(indtheta, indr)] += mask1.astype(bool)
        return mask, mask.sum(0) * self.deltath / (2. * np.pi)

    def lc_mask_with_planet(self, mask, y0p, z0p, rplanet):
        mask = mask.astype(int)
        mask_p, indr_p, indtheta_p = self.create_mask_planet(y0p, z0p, rplanet)
        mask[np.ix_(indtheta_p, indr_p)] = mask_p
        index_r = np.where(mask == 1)[1]
        unique, counts = np.unique(index_r, return_counts=True)
        fraction_spot = np.zeros(len(self.radii))
        fraction_spot[unique] = counts * self.deltath / (2. * np.pi)
        ff_spot = np.sum(fraction_spot * 2. * np.pi *
                         self.radii * self.deltar) / np.pi

        index_r = np.where(mask == self.planet_value)[1]
        unique, counts = np.unique(index_r, return_counts=True)
        fraction_planet = np.zeros(len(self.radii))
        fraction_planet[unique] = counts * self.deltath / (2. * np.pi)
        ff_planet = np.sum(fraction_planet * 2. * np.pi *
                           self.radii*self.deltar) / np.pi
        return fraction_spot, fraction_planet, ff_spot, ff_planet


class OriginalStarModel(BaseStarModel):
    """original Palermo's code wrapped into a class"""
    
    def spot_lc(self, lat, lon, rfeature):
        x0 = np.sin(lat) * np.cos(lon)
        y0 = np.sin(lat) * np.sin(lon)
        z0 = np.cos(lat)

        c = np.sqrt(2.)
        r0 = np.sqrt(y0**2 + z0**2)
        r_min = r0-rfeature
        r_max = r0+rfeature

        value = np.arctan2(z0, y0)
        theta0 = value if value >= 0. else 2.*np.pi+value
        d_theta = c * rfeature / r0
        theta_min = theta0-d_theta
        theta_max = theta0+d_theta

        indr = np.where((self.radii >= r_min) & (self.radii <= r_max))[0]
        if r0 <= rfeature:
            indtheta = np.arange(self.nth, dtype=int)
        else:
            if theta_min >= 0.:
                indtheta = np.where((self.theta >= theta_min) &
                                    (self.theta <= theta_max))[0]
            else:
                theta_min += 2.*np.pi
                indtheta = np.append(np.where(self.theta <= theta_max)[0],
                                     np.where(self.theta >= theta_min)[0])

        dd = np.sqrt((self.X[np.ix_(indtheta, indr)]-x0)**2. +
                     (self.Y[np.ix_(indtheta, indr)]-y0)**2. +
                     (self.Z[np.ix_(indtheta, indr)]-z0)**2.)
        dth = 2.*np.arcsin(dd/2.)
        dth[dth > np.pi/2.] = 0
        dd *= np.cos(dth/2.)
        return dd <= rfeature, indr, indtheta

    def planet_lc(self, y0, z0, rplanet):
        c = np.sqrt(2.)
        r0 = np.sqrt(y0**2 + z0**2)
        r_min = r0 - rplanet
        r_max = r0 + rplanet

        value = np.arctan2(z0, y0)
        theta0 = value if value >= 0 else 2. * np.pi + value
        d_theta = c * rplanet / r0
        theta_min = theta0-d_theta
        theta_max = theta0+d_theta

        indr = np.where((self.radii >= r_min) & (self.radii <= r_max))[0]
        if r0 <= rplanet:
            indtheta = np.arange(self.nth, dtype=int)
        else:
            if theta_min >= 0.:
                indtheta = np.where((self.theta >= theta_min)
                                    & (self.theta <= theta_max))[0]
            else:
                theta_min += 2.*np.pi
                indtheta = np.append(np.where(self.theta <= theta_max)[0],
                                     np.where(self.theta >= theta_min)[0])

        dd = np.sqrt((self.Y[np.ix_(indtheta, indr)]-y0)**2. +
                     (self.Z[np.ix_(indtheta, indr)]-z0)**2.)
        dth = 2. * np.arcsin(dd/2.)
        dth[dth > np.pi/2.] = 0
        dd *= np.cos(dth/2.)
        mask_p = (dd <= rplanet)
        mask_p = mask_p.astype(int)
        mask_p[mask_p == 1] = self.planet_value
        return mask_p, indr, indtheta

    def lc_mask(self, lat, lon, rspot):
        mask = np.zeros([self.nth, self.nr], dtype=bool)

        if isinstance(lat, (int, float)):
            lat = [lat]
        if isinstance(lon, (int, float)):
            lon = [lon]
        if isinstance(rspot, (int, float)):
            rspot = [rspot]
        # print(mask)
        for i in range(len(lat)):
            lat1 = (np.pi / 2. - lat[i] * np.pi / 180.)
            lon1 = lon[i] * np.pi/180.
            mask1, indr, indtheta = self.spot_lc(lat1, lon1, rspot[i])
            mask[np.ix_(indtheta, indr)] += mask1
        return mask, mask.sum(0) * self.deltath / (2.*np.pi)

    def lc_mask_with_planet(self, mask, y0p, z0p, rplanet):
        mask = mask.astype(int)
        mask_p, indr_p, indtheta_p = self.planet_lc(y0p, z0p, rplanet)
        mask[np.ix_(indtheta_p, indr_p)] = mask_p
        index_r = np.where(mask == 1)[1]
        unique, counts = np.unique(index_r, return_counts=True)
        fraction_spot = np.zeros(len(self.radii))
        fraction_spot[unique] = counts * self.deltath / (2. * np.pi)
        ff_spot = np.sum(fraction_spot * 2. * np.pi *
                         self.radii * self.deltar) / np.pi

        index_r = np.where(mask == self.planet_value)[1]
        unique, counts = np.unique(index_r, return_counts=True)
        fraction_planet = np.zeros(len(self.radii))
        fraction_planet[unique] = counts * self.deltath / (2. * np.pi)
        ff_planet = np.sum(fraction_planet * 2. * np.pi *
                           self.radii*self.deltar) / np.pi
        return fraction_spot, fraction_planet, ff_spot, ff_planet
