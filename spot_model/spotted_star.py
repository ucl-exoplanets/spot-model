import warnings
from numbers import Number
from typing import Union, Iterable, Optional

import numpy as np

from spot_model.utils import parse_args_lists, spher_to_cart
from spot_model._base_star import _BaseStar

NumericOrIterable = Optional[Union[Number, Iterable[Number]]]


class SpottedStar(_BaseStar):
    """Star model allowing fast 2D disk integration with spot(s) and transiting planet(s)"""

    def __init__(self, nr: int = 1000,  nth: int = 1000,
                 lat: NumericOrIterable = None,
                 lon: NumericOrIterable = None,
                 rspot: NumericOrIterable = None):
        super().__init__(nr, nth)
        self.spots = {'lat': [],
                      'lon': [],
                      'r': []}

        if lat is not None and lon is not None and rspot is not None:
            self.add_spot(lat, lon, rspot)
        elif lat is not None or lon is not None or rspot is not None:
            raise ValueError(
                "if any of 'lat', 'lon', 'rspot' is specified, these three arguments should be specified too")

    def add_spot(self, lat: NumericOrIterable, lon: NumericOrIterable, rspot: NumericOrIterable):
        lat, lon, rspot = parse_args_lists(lat, lon, rspot)
        self.spots['lat'] += lat
        self.spots['lon'] += lon
        self.spots['r'] += rspot

    def compute_star_mask(self):
        # output mask dimension will be (nt, nr) or (nt, nr, T) or (nt, nr, T, W)
        mask, _ = self._compute_full_spotted_mask(
            self.spots['lat'], self.spots['lon'], self.spots['r'])
        return mask

    def compute_rff(self, yp: Number = None, zp: Number = None, rp: NumericOrIterable = None):
        # Compute the 'observed' radial filling factor
        # output mask dimension will be (nr) or (nr, W) if multiple wavelengths
        mask = self.compute_star_mask()
        if yp is not None and zp is not None and rp is not None:  # occulted situation
            rff_spot, rff_planet, _, _ = self._update_full_mask_with_planet(
                mask, yp, zp, rp)
            return rff_spot, rff_planet
        elif yp is not None or zp is not None or rp is not None:
            raise ValueError(
                "if any of 'yp', 'zp', 'rp' is specified, these three arguments should be specified too")
        else:
            rff = mask.sum(0) * self.deltath / (2. * np.pi)
            return rff

    def compute_ff(self, yp: Number = None, zp: Number = None, rp: NumericOrIterable = None):
        # Compute the 'observed' filling factor
        # (planet independent / out of transit)
        # is ff defined radially or 3D - orginally?

        if yp is not None and zp is not None and rp is not None:  # occulted situation
            rff_spot, rff_planet = self.compute_rff(yp, zp, rp)
            ff_spot = np.sum(rff_spot * 2. * np.pi *
                             self.radii[:, None]*self.deltar, axis=0) / np.pi
            ff_planet = np.sum(rff_planet * 2. * np.pi *
                               self.radii[:, None]*self.deltar, axis=0) / np.pi
            return ff_spot, ff_planet
        elif yp is not None or zp is not None or rp is not None:
            raise ValueError(
                "if any of 'yp', 'zp', 'rp' is specified, these three arguments should be specified too")
        else:
            rff_spot = self.compute_rff(yp, zp, rp)
            ff_spot = np.sum(rff_spot * 2. * np.pi *
                             self.radii*self.deltar, axis=0) / np.pi

            return ff_spot

    def show(self, yp: Number = None, zp: Number = None, rp: NumericOrIterable = None):
        star_mask = self.compute_star_mask()
        return super().show(star_mask, yp, zp, rp)

    ###
    def _create_mask_feat(self, y, z, rfeat, x=None):
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

    def _compute_spot_mask(self, lat, lon, rspot):
        x, y, z = spher_to_cart(lat, lon)
        mask, indr, indth = self._create_mask_feat(y, z, rspot, x=x)
        mask = mask.astype(int)
        if self.spot_value != 1:
            mask[mask.astype(bool)] = self.spot_value
        return mask, indr, indth

    def _compute_planet_mask(self, y, z, rplanet):
        mask, indr, indth = self._create_mask_feat(y, z, rplanet)
        mask = mask.astype(int)
        mask[mask.astype(bool)] = self.planet_value
        return mask, indr, indth

    def _compute_full_spotted_mask(self, lat, lon, rspot):
        mask = np.zeros([self.nth, self.nr], bool)
        if not hasattr(lat, '__len__'):
            lat = [lat]
        if not hasattr(lon, '__len__'):
            lon = [lon]
        if not hasattr(rspot, '__len__'):
            rspot = [rspot]
        for i in range(len(lat)):
            mask1, indr, indtheta = self._compute_spot_mask(
                lat[i], lon[i], rspot[i])
            mask[np.ix_(indtheta, indr)] += mask1.astype(bool)
        return mask.astype(int), mask.sum(0) * self.deltath / (2. * np.pi)

    def _update_full_mask_with_planet(self, mask, y0p, z0p, rplanet):
        if isinstance(rplanet, (int, float)):
            warnings.warn('rplanet expected as array entered as scalar')
            rplanet = np.array([rplanet])
        nw = len(rplanet)

        # planet mask for various radii, and indices of the polar rectangle surrounding the largest radius
        mask_p, indr_p, indtheta_p = self._compute_planet_mask(
            y0p, z0p, rplanet)

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
