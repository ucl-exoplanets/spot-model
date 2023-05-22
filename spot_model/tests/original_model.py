import numpy as np

from spot_model._base_star import _BaseStar


class OriginalStarModel(_BaseStar):
    """original spotted star model code wrapped into a class"""

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
        mask_p[mask_p.astype(bool)] = self.planet_value
        return mask_p, indr, indtheta

    def lc_mask(self, lat, lon, rspot):
        mask = np.zeros([self.nth, self.nr], dtype=bool)

        if not hasattr(lat, '__len__'):
            lat = [lat]
        if not hasattr(lon, '__len__'):
            lon = [lon]
        if not hasattr(rspot, '__len__'):
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
