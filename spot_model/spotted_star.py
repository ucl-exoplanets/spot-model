"""Implement 1D and 2D models of spotted stars.

Classes:

    SpottedStar(_BaseStar)
    SpottedStar1D(SpottedStar)
    SpottedStar2D(SpottedStar)

"""
import warnings
from numbers import Number
from typing import Union, Optional, Tuple

import numpy as np
from numpy import ndarray

from spot_model.utils import parse_args_lists, spher_to_cart, NumOrIt
from spot_model._base_star import _BaseStar


class SpottedStar(_BaseStar):
    """Parent class for spotted stars 1D and 2Ds"""

    def __init__(self,
                 nr: int = 1000,
                 nth: Optional[int] = None,
                 debug: bool = True):
        super().__init__(nr, nth, debug)
        self._spots = {}

    def __repr__(self):
        return f"{self.get_class_name()}(nr={self.nr}, nth={self.nth}, nspots={self.nspots}, debug={self.debug})"

    @property
    def spots(self):
        """return the dictionary of spots"""
        return self._spots

    @property
    def nspots(self) -> int:
        """Return the number of spots."""
        keys = list(self._spots.keys())
        if keys:
            return len(self._spots[keys[0]])
        else:
            return 0

    @property
    def mask(self) -> ndarray:
        """Access the mask for the spotted star. 

        Returns:
            ndarray: 2D int mask for the spotted star with 0 for the plage and 1 for spots.
                First dimension along polar angle and second dimension along polar radius.
        """
        return self._mask

    @property
    def rff(self) -> ndarray:
        """Get the radial spot filling factor of this star.

        Without any occulting planet.
        Returns:
            ndarray: array of dim (nr) ith the spot radial filling factor
        """
        return self.compute_rff()

    @property
    def ff(self) -> Number:
        """Get the spot filling factor of this star.

        Without any occulting planet.
        Returns:
            ndarray: Computed spot filling factor
        """
        return self.compute_ff()

    def add_spot(self):
        """Add one or several spots to the star object."""
        raise NotImplementedError

    def remove_spots(self):
        """Remove all the spots added to the current star"""
        for k in self._spots:
            self._spots[k] = []

        if self._mask is not None and self._mask.any():
            self._mask = np.zeros([self.nth, self.nr])

    def compute_rff(self):
        """Compute the "observed radial filling factor" of the star."""
        raise NotImplementedError

    def compute_ff(self):
        """Compute the "observed filling factor" of the star."""
        raise NotImplementedError


class SpottedStar2D(SpottedStar):
    """Star model allowing fast 2D disk integration with spot(s) and transiting planet(s)"""
    star_value = 0
    spot_value = 1
    planet_value = 2

    def __init__(self, nr: int = 1000,  nth: int = 1000,
                 lat: NumOrIt = None,
                 lon: NumOrIt = None,
                 rspot: NumOrIt = None,
                 debug: bool = True):
        """Create a spotted star object.

        Args:
            nr (int, optional): number of quantised values along polar radius. Defaults to 1000.
            nth (int, optional): number of quantised values along theta (polar angle). 
                (Defaults to 1000)
            lat (NumOrIt, optional): spot(s) latitude(s) in degrees.
                Defaults to None. If defined, must be of same dimension as lon and rspot.
            lon (NumOrIt, optional): spot(s) longitude(s) in degrees. 
                Defaults to None. If defined, must be of same dimension as lat and rspot.
            rspot (NumOrIt, optional): spot(s) radius(es) in degrees.
                Defaults to None. If defined, must be of same dimension as lat and lon.
        """
        super().__init__(nr, nth, debug)
        self._spots = {'lat': [],
                      'lon': [],
                      'r': []}

        self.add_spot(lat, lon, rspot)

    def add_spot(self, lat: NumOrIt, lon: NumOrIt, rspot: NumOrIt):
        """Add one or several spots to the star object.

        Args:
            lat (NumOrIt, optional): spot(s) latitude(s) in degrees.
                Defaults to None. If defined, must be of same dimension as lon and rspot.
            lon (NumOrIt, optional): spot(s) longitude(s) in degrees. 
                Defaults to None. If defined, must be of same dimension as lat and rspot.
            rspot (NumOrIt, optional): spot(s) radius(es) in degrees.
                Defaults to None. If defined, must be of same dimension as lat and lon.
        """
        if lat is not None and lon is not None and rspot is not None:
            lat, lon, rspot = parse_args_lists(
                lat, lon, rspot, same_length=self.debug)
            if self.debug:
                if not (np.greater_equal(lat, -90).all() and np.less_equal(lat, 90).all()):
                    raise ValueError('latitude is defined between -90 and 90°')
                if not (np.greater_equal(lat, -180).all() and np.less_equal(lon, 180).all()):
                    warnings.warn(
                        'longitude is here defined between -180 and 180°')
                if not (np.greater_equal(rspot, 0).all() and np.less_equal(rspot, 1).all()):
                    raise ValueError('rspot should be between 0 and 1')
                if np.isclose(rspot, 0).any():
                    warnings.warn('spot radius is close to zero')
            self._spots['lat'] += lat
            self._spots['lon'] += lon
            self._spots['r'] += rspot
            self._update_mask()

        elif lat is not None or lon is not None or rspot is not None:
            raise ValueError("""if any of 'lat', 'lon', 'rspot' is specified,
                             these three arguments should be specified too""")

    def compute_rff(self,
                    yp: Number = None,
                    zp: Number = None,
                    rp: NumOrIt = None) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Compute the "observed radial filling factor" of the star.

        If no planetary argument is provided (yp, zp, rp), then the method will return the filling
        factor in each annulus defined by the polar grid. If any planetary argument is provided 
        (yp, zp, rp), then all of them should be provided. In that case, the method will also 
        return the observed radial filling factor corresponding to the planet.
        Args:
            yp (Number, optional): planet y position.  Defaults to None.
            zp (Number, optional): planet z position. Defaults to None.
            rp (NumOrIt, optional): planet radius(es). Defaults to None.

        Returns:
            Union[ndarray, Tuple[ndarray, ndarray]]: Either spot radial filling factor (dim=(nr,))
                or tuple with observed spot and planet radial filling factors (dim=(nr, nw)), 
                where nw is the number of planet radii or wavelengths.
        """
        if yp is not None and zp is not None and rp is not None:  # occulted situation
            rff_spot, rff_planet, _, _ = self._update_full_mask_with_planet(
                self.mask, yp, zp, rp)
            return rff_spot, rff_planet
        if yp is not None or zp is not None or rp is not None:
            raise ValueError("""if any of 'yp', 'zp', 'rp' is specified,
                             these three arguments should be specified too""")
        rff = self.mask.sum(0) * self.deltath / (2. * np.pi)
        return rff

    def compute_ff(self,
                   yp: Number = None,
                   zp: Number = None,
                   rp: NumOrIt = None) -> Union[Number, Tuple[NumOrIt, NumOrIt]]:
        """Compute the "observed filling factor" of the star.

        If no planetary argument is provided (yp, zp, rp), then the method will return the filling
        factor of the star as scalar number. If any planetary argument is provided (yp, zp, rp), 
        then all of them should be provided. In that case, the method will also return the 
        observed area ratio of the star occulted by the planet.
        Args:
            yp (Number, optional): planet y position.  Defaults to None.
            zp (Number, optional): planet z position. Defaults to None.
            rp (NumOrIt, optional): planet radius(es). Defaults to None.
        Returns:
            Union[Number, Tuple[NumOrIt, NumOrIt]]: Either spot filling factor
                or tuple with observed spot and planet filling factors (each of length nw), 
                where nw is the number of planet radii or wavelengths. 
        """

        if yp is not None and zp is not None and rp is not None:  # occulted situation
            rff_spot, rff_planet = self.compute_rff(yp, zp, rp)
            if isinstance(rp, Number):
                stellar_radii = self.radii
            else:
                stellar_radii = self.radii[:, None]
            ff_spot = np.sum(rff_spot * 2. * stellar_radii*self.deltar, axis=0)
            ff_planet = np.sum(rff_planet * 2. *
                               stellar_radii*self.deltar, axis=0)
            return ff_spot, ff_planet
        if yp is not None or zp is not None or rp is not None:
            raise ValueError("""if any of 'yp', 'zp', 'rp' is specified,
                             these three arguments should be specified too""")
        rff_spot = self.compute_rff(yp, zp, rp)
        ff_spot = np.sum(rff_spot * 2. * self.radii*self.deltar, axis=0)
        return ff_spot

    #####
    def _update_mask(self):
        """Compute the full 2D mask with defined spot(s).

        TODO: incrementally add to the previously defined mask instead of replacing
        """
        mask, _ = self._compute_full_spotted_mask(
            self._spots['lat'], self._spots['lon'], self._spots['r'])
        self._mask = mask

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

        indr = np.where((self.radii >= r_min) & (self.radii <= r_max))[0]
        if r0 <= rfeatmax:
            indth = np.arange(self.nth, dtype=int)
        else:
            theta0 = np.arctan2(z, y)
            theta0 = theta0 % (2. * np.pi)
            d_theta = np.sqrt(2.) * rfeatmax / (r0+1e-32)
            theta_min = (theta0 - d_theta) % (2. * np.pi)
            theta_max = (theta0 + d_theta) % (2. * np.pi)

            if theta_min <= theta_max:
                indth = np.where((self.theta >= theta_min)
                                 & (self.theta <= theta_max))[0]
            else:
                indth = np.append(np.where(self.theta <= theta_max)[0],
                                  np.where(self.theta >= theta_min)[0])
        dd = np.sqrt(((self.x[np.ix_(indth, indr)] - x)**2. if x else 0) +
                     (self.y[np.ix_(indth, indr)] - y)**2. +
                     (self.z[np.ix_(indth, indr)] - z)**2.)
        dth = 2. * np.arcsin(dd / 2.)
        dth[dth > np.pi / 2.] = 0
        dd *= np.cos(dth / 2.)
        if multir:
            return dd[:, :, None] <= rfeat, indr, indth
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
        for i, la in enumerate(lat):
            mask1, indr, indtheta = self._compute_spot_mask(
                la, lon[i], rspot[i])
            mask[np.ix_(indtheta, indr)] += mask1.astype(bool)
        return mask.astype(int), mask.sum(0) * self.deltath / (2. * np.pi)

    def _update_full_mask_with_planet(self, mask, y0p, z0p, rplanet):
        if isinstance(rplanet, Number):
            stellar_radii = self.radii
            is_scalar_rp = True
            nw = None
        else:
            is_scalar_rp = False
            nw = len(rplanet)
            stellar_radii = self.radii[:, None]

        # planet mask for various radii, & indices of the rectangle surrounding the largest radius
        mask_p, indr_p, indtheta_p = self._compute_planet_mask(
            y0p, z0p, rplanet)

        # planet integration
        if is_scalar_rp:
            fraction_planet = np.zeros(len(self.radii))
        else:
            fraction_planet = np.zeros((len(self.radii), nw))
        fraction_planet[indr_p] = ((mask_p/2).sum(0)*self.deltath)/(2.*np.pi)
        ff_planet = np.sum(fraction_planet * 2. *
                           stellar_radii*self.deltar, axis=0)

        # spot integration
        # Full-size spotted mask with (largest) planet disk removed
        mask_nop = mask.copy()
        mask_nop[np.ix_(indtheta_p, indr_p)] = 0  # Assumes spot_value == 1 !!
        # integration along theta
        fraction_spot = (mask_nop.sum(0) * self.deltath)/(2.*np.pi)  # (nr)
        if not is_scalar_rp:
            fraction_spot = fraction_spot[:, None].repeat(
                nw, axis=1)  # (nr, nw)

        # spotted mask just on the rectangle containing the largest planet disk
        mask_rmax = mask[np.ix_(indtheta_p, indr_p)].astype(int)
        if not is_scalar_rp:
            mask_rmax = mask_rmax[:, :, None].repeat(nw, axis=2)
        mask_rmax *= (~(mask_p.astype(bool))).astype(int)
        # integration along theta
        fraction_spot[indr_p] += (mask_rmax.sum(0)*self.deltath)/(2.*np.pi)
        ff_spot = np.sum(fraction_spot * 2. *
                         stellar_radii * self.deltar, axis=0)
        return fraction_spot, fraction_planet, ff_spot, ff_planet


class SpottedStar1D(SpottedStar):
    """Base class for 1D star model parameterised along radial coordinates, approximating spots as ellipses"""

    def __init__(self, nr: int = 1000,
                 dspot: NumOrIt = None,
                 rspot: NumOrIt = None,
                 debug: bool = True):

        super().__init__(nr=nr, nth=None, debug=debug)

        self._spots = {'d': [],
                      'r': []}
        self.add_spot(dspot, rspot)

    def add_spot(self,
                 dspot: NumOrIt,
                 rspot: NumOrIt):
        """Add one or several spots to the 1D star object.

        Args:
            dspot (NumOrIt, optional): Distance to the centre
                Defaults to None. If defined, must be of same dimension as lon and rspot.
            rspot (NumOrIt, optional): spot(s) radius(es) in degrees.
                Defaults to None. If defined, must be of same dimension as lat and lon.
        """
        if dspot is not None and rspot is not None:
            dspot, rspot = parse_args_lists(
                dspot, rspot, same_length=self.debug)
            if self.debug:
                if not (np.greater_equal(dspot, 0).all() and np.less_equal(dspot, 1).all()):
                    raise ValueError('d is defined between 0 and 1')
                if not (np.greater_equal(rspot, 0).all() and np.less_equal(rspot, 1).all()):
                    raise ValueError('rspot should be between 0 and 1')
                if np.isclose(rspot, 0).any():
                    warnings.warn('spot radius is close to zero')
            self._spots['d'] += dspot
            self._spots['r'] += rspot

            if self.debug and len(self._spots['d']) > 1:
                d_s, r_s = map(
                    list, zip(*sorted(zip(self._spots['d'], self._spots['r']))))
                if (np.array(r_s[:-1]) + np.array(r_s[1:]) > np.diff(d_s)).any():
                    warnings.warn("""Possible overlap between spots in terms of distance,
                                  though the 1D model assumes non-overlapping distances""")

    def compute_rff(self) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Compute the "effective radial filling factor" of the star. 
        Assumes non overlapping and non occulted circular spots with elliptical projections.

        Returns:
            Union[ndarray, Tuple[ndarray, ndarray]]: Either spot radial filling factor (dim=(nr,))
                or tuple with observed spot and planet radial filling factors (dim=(nr, nw)),
                where nw is the number of planet radii or wavelengths.
        """

        rff = np.zeros(self.nr)
        for dspot, rspot in zip(self._spots['d'], self._spots['r']):
            if dspot == 0:
                rff[self.radii <= rspot] += 1
            else:
                mu_spot = np.sqrt(1-dspot**2)  # mu for the spot centre
                beta = rspot * mu_spot  # semi-minor axis of spot ellipse
                z_minus = (1 - mu_spot *
                           np.sqrt(self.mu**2 + rspot**2)) / dspot
                temp = (-mu_spot**2 - self.mu**2 - rspot**2 * mu_spot **
                        2 + 2*mu_spot*np.sqrt(self.mu**2+rspot**2))
                temp[temp < 0] = np.nan
                y_plus = np.sqrt(temp) / dspot
                if dspot >= beta:
                    spotted = np.abs(self.radii - dspot) <= beta
                    rff[spotted] += np.arctan(y_plus /
                                              z_minus)[spotted] / np.pi
                else:
                    case1 = self.radii <= beta - dspot
                    case2 = (np.abs(self.radii - beta) < dspot) & (z_minus > 0)
                    case3 = z_minus == 0
                    case4 = (np.abs(self.radii - beta) < dspot) & (z_minus < 0)
                    # case5 = self.radii >=  beta + dspot  ---> nothing to add in this case
                    rff[case1] += 1
                    rff[case2] += np.arctan(y_plus / z_minus)[case2] / np.pi
                    rff[case3] += 1 / 2
                    rff[case4] += 1 + \
                        np.arctan(y_plus / z_minus)[case4] / np.pi

        if self.debug and (rff > 1).any():
            raise RuntimeError(
                "Radial filling factor can not excede 1, please review the spot parameters")
        return rff

    def compute_ff(self) -> Number:
        """Compute the (effective) filling factor of the star.

        Returns:
            Number: Effective spot filling factor (between 0 and 1)
        """
        rff = self.compute_rff()
        ff = np.sum(rff * 2. * self.radii*self.deltar, axis=0)
        if self.debug and not 0. <= ff <= 1.:
            raise RuntimeError(
                'Filling factor cannot excede 1, please review spot parameters')
        return ff
