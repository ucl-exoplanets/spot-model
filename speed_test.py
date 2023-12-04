"""Perform test speeds to compare new models against the original one.

This script can be execute with: 
> python speed_test.py 

Functions:

    run_speed_test_2d(nr=1000, nth=1000, number=50)
    run_speed_test_1ds(nr=1000, nth=1000, number=50)

"""
import warnings
import timeit

from spot_model.tests.original_model import OriginalStarModel
from spot_model.spotted_star import SpottedStar1D, SpottedStar2D


warnings.filterwarnings('ignore')


def run_speed_test_2d(nr=1000, nth=1000, number=50):
    """Compare 2D model speed with reference model"""
    smodel = SpottedStar2D(nr=nr, nth=nth)
    ref_smodel = OriginalStarModel(nr=nr, nth=nth)

    lat = [0]
    lon = [0]
    rspot = [0.2]
    mask, _ = smodel._compute_full_spotted_mask(lat, lon, rspot)

    # planet radii
    rplanet = 0.05
    y0p = 0.
    z0p = 0.

    # Speed tests
    t0 = timeit.timeit(lambda: ref_smodel.lc_mask(
        lat, lon, rspot), number=number)
    t1 = timeit.timeit(lambda: smodel._compute_full_spotted_mask(
        lat, lon, rspot), number=number)
    print('\n\tTime for two spots and no planet:')
    print(f"ref model: {t0:.3f} s")
    print(f"this model: {t1:.3f} s")

    t0 = timeit.timeit(lambda: ref_smodel.lc_mask_with_planet(
        mask, y0p, z0p, rplanet), number=number)
    t1 = timeit.timeit(lambda: smodel._update_full_mask_with_planet(
        mask, y0p, z0p, rplanet), number=number)
    t2 = timeit.timeit(lambda: smodel._update_full_mask_with_planet(
        mask, y0p, z0p, [rplanet]*10), number=number)
    t3 = timeit.timeit(lambda: smodel._update_full_mask_with_planet(
        mask, y0p, z0p, [rplanet]*100), number=number)
    print('\n\tTime with a planet added:')
    print(f'ref model: {t0:.3f} s')
    print(f"this model: {t1:.3f} s")
    print(f"this model (10 radii): {t2:.3f} s")
    print(f"this model (100 radii): {t3:.3f} s")


def run_speed_test_1d(nr=1000, nth=1000, number=50):
    """Compare 1D model speed with reference model"""
    model2 = SpottedStar2D(nr=nr, nth=nth, lat=0, lon=0, rspot=0.2)
    model1 = SpottedStar1D(nr=nr, dspot=0, rspot=0.2)

    t0 = timeit.timeit(lambda: model1.compute_rff(), number=number)
    t1 = timeit.timeit(lambda: model2.compute_rff(), number=number)

    print('\n\tTime to "compute" the radial filling factor (single central spot):')
    print(f'model1D: {t0:.5f} s')
    print(f"model2D: {t1:.5f} s")

    def f1():
        model1.remove_spots()
        model1.add_spot(dspot=0, rspot=0.2)
        model1.compute_rff()

    def f2():
        model2.remove_spots()
        model2.add_spot(lat=0, lon=0, rspot=0.2)
        model2.compute_rff()

    t0 = timeit.timeit(f1, number=number)
    t1 = timeit.timeit(f2, number=number)

    print('\n\tTime to remove, add spot and compute radial filling factor (single central spot):')
    print(f'model1D: {t0:.5f} s')
    print(f"model2D: {t1:.5f} s")


if __name__ == '__main__':
    run_speed_test_1d()
    run_speed_test_2d()
