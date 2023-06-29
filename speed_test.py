import warnings
import timeit

from spot_model.tests.original_model import OriginalStarModel
from spot_model.spotted_star import SpottedStar2D


warnings.filterwarnings('ignore')


def run_speed_test(nr=1000, nth=1000, number=50):
    smodel = SpottedStar2D(nr=nr, nth=nth)
    ref_smodel = OriginalStarModel(nr=nr, nth=nth)

    lat = 0
    lon = 0
    rspot = 0.2
    mask, _ = smodel._compute_full_spotted_mask(lat, lon, rspot)

    # planet radii
    rplanet = 0.05
    y0p = 0.
    z0p = 0.

    # Speed tests
    t0 = timeit.timeit(lambda:  ref_smodel.lc_mask(
        lat, lon, rspot), number=number)
    t1 = timeit.timeit(lambda:  smodel._compute_full_spotted_mask(
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
    return


if __name__ == '__main__':
    run_speed_test()
