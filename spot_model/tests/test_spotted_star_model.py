import unittest
import logging
from numbers import Number


import numpy as np

from spot_model._base_star import _BaseStar
from spot_model.spotted_star import SpottedStar2D, SpottedStar1D
from spot_model.tests.original_model import OriginalStarModel


EPS = 1e-16


class TestBaseStar(unittest.TestCase):
    def test_polar_grid(self):
        for (nr, nth) in [(10, 10), (20, 50)]:
            star_model = _BaseStar(nr=nr, nth=nth)
            self.assertEqual(star_model.nr, nr)
            self.assertEqual(star_model.nth, nth)
            self.assertEqual(star_model.radii.shape, (nr,))
            self.assertEqual(star_model.mu.shape, (nr,))
            self.assertEqual(star_model.theta.shape, (nth,))
            self.assertEqual(star_model.Z.shape, (nth, nr))

    def test_show(self):
        star_model = _BaseStar(nr=10, nth=10)
        star_model.show()
        star_model.show(yp=0, zp=0, rp=0.05)


class TestSpottedStar2D(unittest.TestCase):
    def test_nospot(self):
        # spots with zero radius
        for lat, lon, rspot in [(None, None, None),  # no spot
                                (0, 0, 0),  # zero radius - central
                                (45, 45, 0),  # zero radius - noncentral
                                (0, 180, 0.5),  # hidden spot behind
                                ]:
            model = SpottedStar2D(lat=lat, lon=lon, rspot=rspot)
            self.assertTrue(np.isclose(model.mask, 0).all())
            rff = model.compute_rff()
            self.assertTrue(np.isclose(rff, 0).all())
            ff = model.compute_ff()
            self.assertTrue(np.isclose(ff, 0).all())

    def test_full_spot(self):
        # Full spot
        model = SpottedStar2D(lat=0, lon=0, rspot=1)
        self.assertTrue(np.isclose(model.mask, 1).all())
        rff = model.compute_rff()
        self.assertTrue(np.isclose(rff, 1).all())
        ff = model.compute_ff()
        self.assertTrue(np.isclose(ff, 1).all())

    def test_one_spot(self):
        model = SpottedStar2D(lat=0, lon=0, rspot=0.2)
        rff = model.compute_rff()
        ff = model.compute_ff()

        model2 = SpottedStar2D(lat=45, lon=45, rspot=0.2)
        rff2 = model2.compute_rff()
        ff2 = model2.compute_ff()

        self.assertTrue(np.greater_equal(model.mask, 0).all()
                        and np.less_equal(model.mask, 1).all())
        self.assertTrue(np.greater_equal(rff, 0).all()
                        and np.less_equal(rff, 1).all())
        self.assertTrue(np.greater_equal(model2.mask, 0).all()
                        and np.less_equal(model2.mask, 1).all())
        self.assertTrue(np.greater_equal(rff2, 0).all()
                        and np.less_equal(rff2, 1).all())
        self.assertTrue(0 < ff2 < ff < 1)

        # compatibility with original model
        ref_model = OriginalStarModel()
        ref_mask, ref_rff = ref_model.lc_mask(0, 0, 0.2)

        self.assertTrue(np.isclose(rff, ref_rff).all())
        self.assertTrue(np.isclose(model.mask, ref_mask).all())

        # symmetry wrt X axis
        model1 = SpottedStar2D(lat=-0.05, lon=20, rspot=0.15)
        model2 = SpottedStar2D(lat=0.05, lon=20, rspot=0.15)
        self.assertTrue(np.isclose(model1.ff, model2.ff))

    def test_wrong_spot(self):
        for rspot in [-0.5, 1.5]:
            model = SpottedStar2D()
            self.assertRaises(ValueError, lambda: model.add_spot(0, 0, rspot))

        for lat in [-91, 93]:
            model = SpottedStar2D()
            self.assertRaises(ValueError, lambda: model.add_spot(lat, 0, 0.1))

    def test_multiple_spots(self):
        # non overlapping
        model = SpottedStar2D(lat=[0, 45], lon=[0, 45], rspot=[0.1, 0.1])
        rff = model.compute_rff()
        ff = model.compute_ff()
        model1 = SpottedStar2D(lat=0, lon=0, rspot=0.1)
        rff1 = model1.compute_rff()
        ff1 = model1.compute_ff()
        model2 = SpottedStar2D(lat=45, lon=45, rspot=0.1)
        rff2 = model2.compute_rff()
        ff2 = model2.compute_ff()

        self.assertTrue(np.isclose(model.mask, model1.mask+model2.mask).all())
        # ok cause disjoint in radius too
        self.assertTrue(np.isclose(rff, rff1+rff2).all())
        self.assertTrue(np.isclose(ff, ff1+ff2).all())

        # fully overlapping
        model = SpottedStar2D(lat=[0, 0], lon=[0, 0], rspot=[0.05, 0.1])
        rff = model.compute_rff()
        ff = model.compute_ff()

        self.assertTrue(np.isclose(model.mask, model1.mask).all())
        self.assertTrue(np.isclose(rff, rff1).all())
        self.assertTrue(np.isclose(ff, ff1).all())

        # partially overlapping
        model = SpottedStar2D(lat=[-5, 5], lon=[2, 5], rspot=[0.2, 0.2])
        rff = model.compute_rff()
        ff = model.compute_ff()
        model1 = SpottedStar2D(lat=-5, lon=2, rspot=0.2)
        rff1 = model1.compute_rff()
        ff1 = model1.compute_ff()
        model2 = SpottedStar2D(lat=5, lon=5, rspot=0.2)
        rff2 = model2.compute_rff()
        ff2 = model2.compute_ff()

        self.assertTrue(np.less_equal(
            model.mask, model1.mask+model2.mask).all())
        self.assertTrue(np.less_equal(rff, rff1+rff2+EPS).all())
        self.assertTrue(np.less(ff, ff1+ff2).all())

    def test_one_spot_and_monochrome_planet(self):

        # Central spot and planet
        fixtures = {  # planet inside central spot
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

            model = SpottedStar2D(
                lat=kwargs['lat'], lon=kwargs['lon'], rspot=kwargs['rspot'])
            rff_spot_noplanet = model.compute_rff()
            ff_spot_noplanet = model.compute_ff()
            rff_spot, rff_planet = model.compute_rff(
                yp=kwargs['y0p'], zp=kwargs['z0p'], rp=kwargs['rplanet'])
            ff_spot, ff_planet = model.compute_ff(
                yp=kwargs['y0p'], zp=kwargs['z0p'], rp=kwargs['rplanet'])

            rff_spot_occulted = rff_spot_noplanet - rff_spot.squeeze()
            
            # shapes
            self.assertEqual(rff_spot.shape, (model.nr, ))
            self.assertEqual(rff_planet.shape, (model.nr, ))
            self.assertTrue(isinstance(ff_spot, Number))
            self.assertTrue(isinstance(ff_planet, Number))
            
            # basic physics
            self.assertTrue(np.less_equal(rff_spot_occulted,
                            rff_spot_noplanet+EPS).all())
            self.assertTrue(np.less_equal(rff_spot_occulted,
                            rff_planet.squeeze()+EPS).all())

            # geometric logic
            if k in [0, 4]:
                self.assertTrue(np.isclose(rff_spot_occulted,
                                rff_planet.squeeze()).all())
                # be wary of dimensioality
                self.assertTrue((0 < ff_planet < ff_spot).all())

            if k in [1, 5]:
                self.assertTrue(np.isclose(rff_spot_occulted, 0).all())
                # be wary of dimensioality
                self.assertTrue(np.isclose(ff_spot, ff_spot_noplanet).all())
            if k in [3, 7]:
                self.assertTrue(np.isclose(rff_spot_occulted,
                                rff_spot_noplanet).all())
                # be wary of dimensioality
                self.assertTrue((ff_spot < ff_spot_noplanet < ff_planet).all())
            if k in [2, 6]:
                self.assertTrue(np.less_equal(
                    rff_spot_noplanet, (rff_spot+rff_planet).squeeze() + EPS).all())
                # be wary of dimensioality
                self.assertTrue((0 < ff_spot < ff_spot_noplanet).all())

    def test_polychrome_planet(self):
        model = SpottedStar2D()
        spot_rff, planet_rff = model.compute_rff(0, 0, [0.1, 0.1])
        spot_ff, planet_ff = model.compute_ff(0, 0, [0.1, 0.1])
        self.assertEqual(spot_rff.shape, (model.nr, 2))
        self.assertEqual(planet_rff.shape, (model.nr, 2))
        self.assertEqual(spot_ff.shape, (2,))
        self.assertEqual(planet_ff.shape, (2,))


class TestSpottedStar1D(unittest.TestCase):
    def test_nospot(self):
        # spots with zero radius
        for dspot, rspot in [(None, None),  # no spot
                             (0, 0),  # zero radius - central
                             (0.5, 0)  # zero radius - noncentral
                             ]:
            model = SpottedStar1D(dspot=dspot, rspot=rspot)
            rff = model.compute_rff()
            self.assertTrue(np.isclose(rff, 0).all())
            ff = model.compute_ff()
            self.assertTrue(np.isclose(ff, 0).all())

    def test_full_spot(self):
        # Full spot
        model = SpottedStar1D(dspot=0, rspot=1)
        rff = model.compute_rff()
        self.assertTrue(np.isclose(rff, 1).all())
        ff = model.compute_ff()
        self.assertTrue(np.isclose(ff, 1).all())

    def test_one_spot(self):
        model = SpottedStar1D(dspot=0, rspot=0.2)
        rff = model.compute_rff()
        ff = model.compute_ff()

        model2 = SpottedStar1D(dspot=0.5, rspot=0.2)
        rff2 = model2.compute_rff()
        ff2 = model2.compute_ff()

        self.assertTrue(np.greater_equal(rff, 0).all()
                        and np.less_equal(rff, 1).all())
        self.assertTrue(np.greater_equal(rff2, 0).all()
                        and np.less_equal(rff2, 1).all())
        self.assertTrue(0 < ff2 < ff < 1)

        # compatibility with 2D model 
        # (centre spot)
        ref_model = SpottedStar2D(lat=0, lon=0, rspot=0.2)
        self.assertTrue(np.isclose(rff, ref_model.rff).all())
        self.assertTrue(np.isclose(ff, ref_model.ff).all())

        # non central spot - discrepancy to investigate (see issue #31)
        ref_model2 = SpottedStar2D(lat=30, lon=0, rspot=0.2, nth=10000)
        self.assertTrue(np.isclose(rff2, ref_model2.rff).all())
        self.assertTrue(np.isclose(ff2, ref_model2.ff).all())
        
    def test_wrong_spot(self):
        for rspot in [-0.5, 1.5]:
            model = SpottedStar1D()
            self.assertRaises(ValueError, lambda: model.add_spot(0.1, rspot))

        for dspot in [-0.2, 1.1]:
            model = SpottedStar1D()
            self.assertRaises(ValueError, lambda: model.add_spot(dspot, 0.1))

    def test_multiple_spots(self):
        # non overlapping
        model = SpottedStar1D(dspot=[0, 0.5], rspot=[0.1, 0.1])
        rff = model.compute_rff()
        ff = model.compute_ff()
        model1 = SpottedStar1D(dspot=0, rspot=0.1)
        rff1 = model1.compute_rff()
        ff1 = model1.compute_ff()
        model2 = SpottedStar1D(dspot=0.5, rspot=0.1)
        rff2 = model2.compute_rff()
        ff2 = model2.compute_ff()

        # ok cause disjoint in radius too
        self.assertTrue(np.isclose(rff, rff1+rff2).all())
        self.assertTrue(np.isclose(ff, ff1+ff2).all())

if __name__ == '__main__':
    unittest.main()
