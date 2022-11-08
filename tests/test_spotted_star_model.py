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
        for lat in np.array([0, np.pi/2]):
            for lon in [0, np.pi/2]:
                for rspot in [0., 0.1, 0.5]:
                    mask, ff = smodel.lc_mask(lat, lon, rspot)
                    mask0, ff0 = ref_smodel.lc_mask(lat *180/np.pi, lon*180/np.pi, rspot)
                    try:
                        self.assertTrue(np.isclose(ff, ff0).all())
                        self.assertTrue(np.isclose(mask, mask0).all())
                    except AssertionError as e:
                        logging.info(
                            f"\n Error at: lon={lon}, lat={lat}")# --> ff0 = {ff0}, ff = {ff}")
                        raise e

        # Speed test
        t0 = timeit.timeit(lambda:  ref_smodel.lc_mask(0, 0, 0.1), number=50)
        t1 = timeit.timeit(lambda:  smodel.lc_mask(0, 0, 0.1), number=50)
        logging.info(f"\n Palermo's execution time: {t0} ")
        logging.info(f"\n This code's execution time: {t1} ")

        # Multiple spots
        lat = np.array([0, np.pi/4])
        lon = np.array([0, np.pi/4])
        rspot = np.array([0.2, 0.1])
        mask, _ = smodel.lc_mask(lat, lon, rspot)
        mask0, _ = ref_smodel.lc_mask(lat*180/np.pi, lon*180/np.pi, rspot)
        self.assertTrue(np.isclose(1+mask, 1+mask0).all())

        # Speed tests
        t0 = timeit.timeit(lambda:  ref_smodel.lc_mask(
            lat, lon, rspot), number=50)
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi
        t1 = timeit.timeit(lambda:  smodel.lc_mask(lat, lon, rspot), number=50)
        logging.info(f"\n Palermo's execution time (2 spots): {t0} ")
        logging.info(f"\n This code's execution time (2 spots): {t1} ")
        #self.assertLess(t1, t0)

    def test_planet_mask(self):
        ...

    def test_mask(self):
        ...


if __name__ == '__main__':
    unittest.main()
