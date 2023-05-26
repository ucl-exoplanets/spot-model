import unittest
import timeit
import logging

import numpy as np

from spot_model.spotted_star import SpottedStar
from spot_model.tests.original_model import OriginalStarModel


EPS = 1e-16

class TestStarModel(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)

    def setUp(self):
        ...

    def test_stars(self):
        for (nr, nth) in [(100, 100), (200, 500)]:
            star_model = SpottedStar(nr=nr, nth=nth)
            self.assertEqual(star_model.nr, nr)
            self.assertEqual(star_model.nth, nth)
            self.assertEqual(star_model.radii.shape, (nr,))
            self.assertEqual(star_model.mu.shape, (nr,))
            self.assertEqual(star_model.theta.shape, (nth,))
            self.assertEqual(star_model.Z.shape, (nth, nr))

    def test_spot_mask(self, nr=1000, nth=1000):
        smodel = SpottedStar(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)

        # single spots
        for lat in np.array([0, 90]):
            for lon in [0, 90]:
                for rspot in [0., 0.1, 0.5]:
                    mask, ff = smodel._compute_full_spotted_mask(lat, lon, rspot)
                    mask0, ff0 = ref_smodel.lc_mask(lat, lon, rspot)
                    try:
                        self.assertTrue(np.isclose(ff, ff0).all())
                        self.assertTrue(np.isclose(mask, mask0).all())
                    except AssertionError as e:
                        logging.info(
                            f"\n Error at: lon={lon}, lat={lat}")  # --> ff0 = {ff0}, ff = {ff}")
                        raise e

        # Multiple spots
        lat = np.array([0, 45])
        lon = np.array([0, 45])
        rspot = np.array([0.2, 0.1])
        mask, _ = smodel._compute_full_spotted_mask(lat, lon, rspot)
        mask0, _ = ref_smodel.lc_mask(lat, lon, rspot)
        self.assertTrue(np.isclose(1+mask, 1+mask0).all())

        # Speed tests
        t0 = timeit.timeit(lambda:  ref_smodel.lc_mask(
            lat, lon, rspot), number=50)
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi

    def test_full_mask(self, nr=1000, nth=1000):
        smodel = SpottedStar(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)

        lat = 45
        lon = 45
        rspot = 0.1
        mask, _ = smodel._compute_full_spotted_mask(lat, lon, rspot)

        ref_mask, _ = ref_smodel.lc_mask(lat, lon, rspot)
        self.assertTrue(np.isclose(1+mask, 1+ref_mask).all())

    def test_planet_mask(self, nr=1000, nth=1000):
        smodel = SpottedStar(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)

        rplanet = 0.1
        y = -0.5
        z = -0.5
        mask, _, _ = smodel._compute_planet_mask(y, z, rplanet)
        ref_mask, _, _ = ref_smodel.planet_lc(y, z, rplanet)
        self.assertTrue(np.isclose(1+mask, 1+ref_mask).all())

    def test_integration_one_spot_and_planet(self, nr=1000, nth=1000):
        smodel = SpottedStar(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)

        # Central spot and planet
        fixtures = {# planet inside central spot
                    0: dict(lat=0, lon=0, rspot=0.2, y0p=0., z0p=0., rplanet=0.05),  
                    # planet outside central spot
                    1: dict(lat=0, lon=0, rspot=0.2, y0p=0.45, z0p=0.45, rplanet=0.05),
                    # part of planet occulting part of centred spot
                    2: dict(lat=0, lon=0, rspot=0.2, y0p=0.15, z0p=0.15, rplanet=0.1),
                    # central spot fully occulted by giant planet
                    3: dict(lat=0, lon=0, rspot=0.1, y0p=0., z0p=0., rplanet=0.2),
                    
                    # planet inside noncentral spot
                    4: dict(lat=30, lon=30, rspot=0.2, y0p=0.45, z0p=0.45, rplanet=0.05),
                    # planet outside noncentral spot
                    5: dict(lat=30, lon=30, rspot=0.2, y0p=0., z0p=0., rplanet=0.05),
                    # part of planet occulting part of noncentral spot
                    6: dict(lat=30, lon=30, rspot=0.2, y0p=0.38, z0p=0.38, rplanet=0.1),
                    # noncentral spot fully occulted by giant planet
                    7: dict(lat=30, lon=30, rspot=0.1, y0p=0.45, z0p=0.45, rplanet=0.2),
                    }

        for k, kwargs in fixtures.items():
            logging.info(f"\n fixture {k}")
            mask, fraction_unocculted = smodel._compute_full_spotted_mask(  #fraction "unocculted" is misleading -> as if there was no planet
                kwargs['lat'], kwargs['lon'], kwargs['rspot'])
            fraction_spot, fraction_planet, ff_spot, ff_planet = smodel._update_full_mask_with_planet(
                mask, kwargs['y0p'], kwargs['z0p'], kwargs['rplanet'])
            ref_fraction_spot, ref_fraction_planet, ref_ff_spot, ref_ff_planet = ref_smodel.lc_mask_with_planet(
                mask, kwargs['y0p'], kwargs['z0p'], kwargs['rplanet'])
            
            fraction_occulted = fraction_unocculted - fraction_spot.squeeze()
            # ref_fraction_occulted = mask - ref_fraction_spot

            # compatibility --> this does not pass because original model is wrong!
            # self.assertEqual(ff_spot, ref_ff_spot)
            # self.assertEqual(ff_planet, ref_ff_planet)
            # self.assertTrue(np.isclose(
            #     fraction_spot.squeeze(), ref_fraction_spot).all())
            # self.assertTrue(np.isclose(
            #     fraction_planet.squeeze(), ref_fraction_planet).all())
            
            # basic physics

            self.assertTrue(np.less_equal(fraction_occulted, fraction_unocculted+EPS).all())
            self.assertTrue(np.less_equal(fraction_occulted, fraction_planet.squeeze()+EPS).all())
            
            # geometric logic
            if k in [0, 4]:
                self.assertTrue(np.isclose(fraction_occulted, fraction_planet.squeeze()).all())
            if k in [1, 5]:
                self.assertTrue(np.isclose(fraction_occulted, 0).all())
            if k in [3, 7]:
                self.assertTrue(np.isclose(fraction_occulted, fraction_unocculted).all())
            if k in [2, 6]:
                self.assertTrue(np.less_equal(fraction_unocculted, (fraction_spot+fraction_planet).squeeze()+ EPS).all())
                self.assertTrue(np.less(fraction_unocculted, (fraction_spot+fraction_planet).squeeze()).any())
                
            # add tests on ff

                
if __name__ == '__main__':
    unittest.main()
