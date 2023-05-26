# spot-model

Star model for spot and transit computation. 

This code solely focuses on the geometric distribution of spot(s) and planet(s) on the disk, and thus doesn't include any spectral or limb-darkening model.

It commonly models the observed star as a 2D stellar disk by discretising it in polar coordinates, computing the corresponding mask indicating the presence of spot(s) and planet(s), and integrating it along the two polar coordinates. 

Furthermore, this model assumes that stars and planets are spherical and spots are spherical caps of identical contrast.

Basic use:
```python
from spot_model import SpottedStar

# define star model
model = SpottedStar(lat=[30, -13], lon=[-45,20],rspot=[0.1, 0.2])

# display star
model.show()

# gets filling factor
print('filling factor:', model.ff)

# add planet and compute radial filling factors
model.compute_rff(yp=0., zp=0., rp=0.1)
```
Some more examples in [quick_start.ipynb](https://github.com/ucl-exoplanets/spot-model/blob/main/quick_start.ipynb).

This work stems from Ariel stellar activity working group, building on the model originally developed in Palermo's group.

Any question please get in touch (mario.morvan@ucl.ac.uk), [raise an issue](https://github.com/ucl-exoplanets/spot-model/issues) or [create a PR](https://github.com/ucl-exoplanets/spot-model/pulls). 