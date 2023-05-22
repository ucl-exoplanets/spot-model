import timeit

from spot_model.spotted_star import SpottedStar, OriginalStarModel


def run_speed_test(nr=1000, nth=1000, number=50):
    smodel = SpottedStar(nr=nr, nth=nth)
    ref_smodel = OriginalStarModel(nr=nr, nth=nth)
    
    lat = 0
    lon = 0
    rspot = 0.2
    mask, _ = smodel.lc_mask(lat, lon, rspot)

    # planet radii
    rplanet = 0.05 
    y0p = 0.
    z0p = 0.
    
    # Speed tests
    t0 = timeit.timeit(lambda:  ref_smodel.lc_mask(lat, lon, rspot), number=number)
    t1 = timeit.timeit(lambda:  smodel.lc_mask(lat, lon, rspot), number=number)
    print(f"\n ref execution time (2 spots): {t0} ")
    print(f"\n This code's execution time (2 spots): {t1} ")

    
    t0 = timeit.timeit(lambda: ref_smodel.lc_mask_with_planet(mask, y0p, z0p, rplanet), number=number)
    t1 = timeit.timeit(lambda: smodel.lc_mask_with_planet(mask, y0p, z0p, rplanet), number=number)
    t2 = timeit.timeit(lambda: smodel.lc_mask_with_planet(mask, y0p, z0p, [rplanet]*10), number=number)
    t3 = timeit.timeit(lambda: smodel.lc_mask_with_planet(mask, y0p, z0p, [rplanet]*100), number=number)
    
    print('ref time:', t0)
    print("this code's time:", t1)
    print("this code's time (10 radii):", t2)
    print("this code's time (100 radii):", t3)
    return
    
if __name__=='__main__':
    run_speed_test()