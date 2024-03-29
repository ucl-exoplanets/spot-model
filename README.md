# Spot-Model

Simple model of a spotted star with transiting planet(s). 

## *Description*

This code solely focuses on the geometric distribution of spot(s) and planet(s) on the disk, and thus doesn't include any spectral or limb-darkening information as such.

It models the observed star as a 2D stellar disk by discretising it in polar coordinates, computing the corresponding mask indicating the presence of spot(s) and planet(s), and integrating it along the two polar coordinates. Behind the hood, this model assumes that stars and planets are spherical and spots are spherical caps of identical contrast.

Alternatively, there is also a 1D spotted star model to compute more efficiently the radial profile of a spotted star, but note that this model approximates spherical caps projections as ellipses and can't accommodate for transit calculations.


## *Basic use*

```python
from spot_model import SpottedStar2D

# define a star with two spots
model = SpottedStar2D(lat=[30, -13], lon=[-45,20], rspot=[0.1, 0.2])

# display it
model.show()

# access the filling factor
print('filling factor:', model.ff)

# add a planet and compute the observed radial filling factors
model.compute_rff(yp=0., zp=0., rp=0.1)

```

See [quick_start.ipynb](https://github.com/ucl-exoplanets/spot-model/blob/main/quick_start.ipynb) for a more detailed walkthrough.

## *Credits and license*

This work stems from the [Ariel](https://arielmission.space/) stellar activity working group, and builds on the spot model originally developed in Cracchiolo et al. ([2021a](https://arxiv.org/abs/2108.12526), [2021b](https://arxiv.org/abs/2108.12526)).

Any question, issue or suggestion? Please feel free to [raise an issue](https://github.com/ucl-exoplanets/spot-model/issues), [create a PR](https://github.com/ucl-exoplanets/spot-model/pulls) or [get in touch via email](mario.morvan@ucl.ac.uk).

Spot-Model is licensed under GPL3, see the [full license here](https://github.com/ucl-exoplanets/spot-model/blob/main/LICENSE.md).