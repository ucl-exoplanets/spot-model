import unittest
import timeit
import logging

import numpy as np

from models.spotted_star import StarModel, OriginalStarModel


class TestStarModel(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)

    def setUp(self):
        ...

    def test_stars(self):
        for (nr, nth) in [(100, 100), (200, 500)]:
            star_model = StarModel(nr=nr, nth=nth)
            self.assertEqual(star_model.nr, nr)
            self.assertEqual(star_model.nth, nth)
            self.assertEqual(star_model.radii.shape, (nr,))
            self.assertEqual(star_model.mu.shape, (nr,))
            self.assertEqual(star_model.theta.shape, (nth,))
            self.assertEqual(star_model.Z.shape, (nth, nr))

    def test_spot_mask(self, nr=1000, nth=1000):
        smodel = StarModel(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)

        # single spots
        for lat in np.array([0, 90]):
            for lon in [0, 90]:
                for rspot in [0., 0.1, 0.5]:
                    mask, ff = smodel.lc_mask(lat, lon, rspot)
                    mask0, ff0 = ref_smodel.lc_mask(lat, lon, rspot)
                    try:
                        self.assertTrue(np.isclose(ff, ff0).all())
                        self.assertTrue(np.isclose(mask, mask0).all())
                    except AssertionError as e:
                        logging.info(
                            f"\n Error at: lon={lon}, lat={lat}")# --> ff0 = {ff0}, ff = {ff}")
                        raise e

        # Multiple spots
        lat = np.array([0, 45])
        lon = np.array([0, 45])
        rspot = np.array([0.2, 0.1])
        mask, _ = smodel.lc_mask(lat, lon, rspot)
        mask0, _ = ref_smodel.lc_mask(lat, lon, rspot)
        self.assertTrue(np.isclose(1+mask, 1+mask0).all())

        # Speed tests
        t0 = timeit.timeit(lambda:  ref_smodel.lc_mask(
            lat, lon, rspot), number=50)
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi


    def test_full_mask(self, nr=1000, nth=1000):
        smodel = StarModel(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)
        
        lat = 45
        lon = 45
        rspot = 0.1
        mask, _ = smodel.lc_mask(lat, lon, rspot)
        
        ref_mask,_ = ref_smodel.lc_mask(lat, lon, rspot)
        self.assertTrue(np.isclose(1+mask, 1+ref_mask).all())

    def test_planet_mask(self, nr=1000, nth=1000):
        smodel = StarModel(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)
        
        rplanet = 0.1  #np.array([0.2, 0.1])
        y = -0.5
        z = -0.5
        mask, _, _ = smodel.create_mask_planet(y, z, rplanet)
        ref_mask, _, _ = ref_smodel.planet_lc(y, z, rplanet)
        self.assertTrue(np.isclose(1+mask, 1+ref_mask).all())

    def test_integration(self, nr=1000, nth=1000):
        smodel = StarModel(nr=nr, nth=nth)
        ref_smodel = OriginalStarModel(nr=nr, nth=nth)
        
        lat = 0
        lon = 0
        rspot = 0.2
        mask, _ = smodel.lc_mask(lat, lon, rspot)
        
        # planet radii
        rplanet = 0.05  #np.array([0.2, 0.1])
        y0p = 0.
        z0p = 0.
        result = smodel.lc_mask_with_planet(mask, y0p, z0p, rplanet)
        ref_result = ref_smodel.lc_mask_with_planet(mask, y0p, z0p, rplanet)
        self.assertEqual(result[-2], ref_result[-2])
        self.assertEqual(result[-1], ref_result[-1])
        self.assertTrue(np.isclose(result[0].flatten(), ref_result[0]).all())
        self.assertTrue(np.isclose(result[1].flatten(), ref_result[1]).all())


if __name__ == '__main__':
    unittest.main()
