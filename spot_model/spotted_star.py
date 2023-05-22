import warnings

import numpy as np

from spot_model.utils import spher_to_cart
from spot_model._base_star import _BaseStar


class SpottedStar(_BaseStar):
    def create_mask_feat(self, y, z, rfeat, x=None):
        r0 = np.sqrt(y**2 + z**2)

        if isinstance(rfeat, (int, float)):
            multir = False
            rfeatmax = rfeat
        else:
            multir = True
            rfeat = np.array(rfeat)
            rfeatmax = np.max(rfeat)

        r_min = r0 - rfeatmax
        r_max = r0 + rfeatmax

        theta0 = np.arctan2(z, y)
        theta0 = theta0 % (2. * np.pi)
        d_theta = np.sqrt(2.) * rfeatmax / r0
        theta_min = theta0 - d_theta
        theta_max = theta0 + d_theta

        indr = np.where((self.radii >= r_min) & (self.radii <= r_max))[0]
        if r0 <= rfeatmax:
            indth = np.arange(self.nth, dtype=int)
        else:
            if theta_min >= 0.:
                indth = np.where((self.theta >= theta_min)
                                 & (self.theta <= theta_max))[0]
            else:
                theta_min += 2. * np.pi
                indth = np.append(np.where(self.theta <= theta_max)[0],
                                  np.where(self.theta >= theta_min)[0])
        dd = np.sqrt(((self.X[np.ix_(indth, indr)] - x)**2. if x else 0) +
                     (self.Y[np.ix_(indth, indr)] - y)**2. +
                     (self.Z[np.ix_(indth, indr)] - z)**2.)
        dth = 2. * np.arcsin(dd / 2.)
        dth[dth > np.pi / 2.] = 0
        dd *= np.cos(dth / 2.)
        if multir:
            return dd[:, :, None] <= rfeat, indr, indth
        else:
            return dd <= rfeat, indr, indth

    def create_mask_spot(self, lat, lon, rspot):
        x, y, z = spher_to_cart(lat, lon)
        mask, indr, indth = self.create_mask_feat(y, z, rspot, x=x)
        mask = mask.astype(int)
        if self.spot_value != 1:
            mask[mask.astype(bool)] = self.spot_value
        return mask, indr, indth

    def create_mask_planet(self, y, z, rplanet):
        mask, indr, indth = self.create_mask_feat(y, z, rplanet)
        mask = mask.astype(int)
        mask[mask.astype(bool)] = self.planet_value
        return mask, indr, indth

    def lc_mask(self, lat, lon, rspot):
        mask = np.zeros([self.nth, self.nr], bool)
        if not hasattr(lat, '__len__'):
            lat = [lat]
        if not hasattr(lon, '__len__'):
            lon = [lon]
        if not hasattr(rspot, '__len__'):
            rspot = [rspot]
        for i in range(len(lat)):
            mask1, indr, indtheta = self.create_mask_spot(
                lat[i], lon[i], rspot[i])
            mask[np.ix_(indtheta, indr)] += mask1.astype(bool)
        return mask.astype(int), mask.sum(0) * self.deltath / (2. * np.pi)

    def lc_mask_with_planet(self, mask, y0p, z0p, rplanet):
        if isinstance(rplanet, (int, float)):
            warnings.warn('rplanet expected as array entered as scalar')
            rplanet = np.array([rplanet])
        nw = len(rplanet)

        # planet mask for various radii, and indices of the polar rectangle surrounding the largest radius
        mask_p, indr_p, indtheta_p = self.create_mask_planet(y0p, z0p, rplanet)

        # planet integration
        fraction_planet = np.zeros((len(self.radii), nw))
        fraction_planet[indr_p] = ((mask_p/2).sum(0)*self.deltath)/(2.*np.pi)
        ff_planet = np.sum(fraction_planet * 2. * np.pi *
                           self.radii[:, None]*self.deltar, axis=0) / np.pi

        # spot integration

        # Full-size spotted mask with (largest) planet disk removed
        mask_nop = mask.copy()
        mask_nop[np.ix_(indtheta_p, indr_p)] = 0  # Assumes spot_value == 1 !!
        # integration along theta
        fraction_spot = (mask_nop.sum(0) * self.deltath)/(2.*np.pi)  # (nr)
        fraction_spot = fraction_spot[:, None].repeat(nw, axis=1)  # (nr, nw)

        # spotted mask just on the rectangle containing the largest planet disk
        mask_rmax = mask[np.ix_(indtheta_p, indr_p)]
        mask_rmax = mask_rmax[:, :, None].repeat(nw, axis=2).astype(int)
        mask_rmax *= (~(mask_p.astype(bool))).astype(int)
        # integration along theta
        fraction_spot[indr_p] += (mask_rmax.sum(0)*self.deltath)/(2.*np.pi)
        ff_spot = np.sum(fraction_spot * 2. * np.pi *
                         self.radii[:, None] * self.deltar, axis=0) / np.pi
        return fraction_spot, fraction_planet, ff_spot, ff_planet