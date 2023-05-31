# spot-model

Simple star model for spot and transit computation. 

This code solely focuses on the geometric distribution of spot(s) and planet(s) on the disk, and thus doesn't include any spectral or limb-darkening model.

It commonly models the observed star as a 2D stellar disk by discretising it in polar coordinates, computing the corresponding mask indicating the presence of spot(s) and planet(s), and integrating it along the two polar coordinates. 

Furthermore, this model assumes that stars and planets are spherical and spots are spherical caps of identical contrast.

Basic use:
```python
from spot_model import SpottedStar

# define a star with two spots
model = SpottedStar(lat=[30, -13], lon=[-45,20],rspot=[0.1, 0.2])

# display it
model.show()

# access the filling factor
print('filling factor:', model.ff)

# add a planet and compute the observed radial filling factors
model.compute_rff(yp=0., zp=0., rp=0.1)
```

See [quick_start.ipynb](https://github.com/ucl-exoplanets/spot-model/blob/main/quick_start.ipynb) for a more detailed walkthrough.

This work stems from the [Ariel](https://arielmission.space/) stellar activity working group, and builds on the spot model originally developed in Cracchiolo et al. ([2021a](https://arxiv.org/abs/2108.12526), [2021b](https://arxiv.org/abs/2108.12526)).

Any question? Please get in touch (mario.morvan@ucl.ac.uk), [raise an issue](https://github.com/ucl-exoplanets/spot-model/issues) or [create a PR](https://github.com/ucl-exoplanets/spot-model/pulls).

This code is licensed under GPL3, see LICENSE.md for the full license file.