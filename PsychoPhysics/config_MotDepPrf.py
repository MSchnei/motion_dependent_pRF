# -*- coding: utf-8 -*-
"""Define stimulus parameters for motion dependent pRF"""

import numpy as np

# %% general stimulus parameters

# number of expected frames per second
nFrames = 60.
# number of pixels for the entire filed of view
pix = 1024.
# height of the entire field of view in deg of vis angle
fovHeight = 20.4
# size of the wedge-ring aperture radial direction in deg of vis angle
barSize = 1.7
# border range for raised cosine ind eg of visual angle
borderRange = 0.6
# how many apertures make one circle?
numAprtCrcle = 6.
# what should be the angle of the ring-wedge
wedgeAngle = 120.

# define opacity on/off cycle in ms
# for stimulus
lenCycStim = 300.
# for raised cosine
lenCycRamp = 50.
# for blank rest
lenCycRest = 500.
# for initial period
lenCycInit = 400.

# %% set parameters for the texture

# define the spatial frequency of the radial sine wave grating
spatFreq = 1.
# define the number of angular cycles of the radial sine wave grating
angularCycles = np.ceil(np.pi*fovHeight /
                        ((fovHeight/2.) / ((fovHeight/2)*spatFreq*2))
                        )
# number of cycles per deg of vis angle per second
cycPerSec = 5.

# %%
combis = np.array([[3, 2], [2, 3], [2, 2], [3, 3], [3, 1], [1, 3], [2, 1],
                   [1, 2]])

posShifts = np.array([-0.8, -0.4, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.4,
                      0.8])

ecc = np.array([4.2, 7.7])

positions = posShifts[:, None] + ecc[None, :]

eccPositions = np.where(positions == ecc)[0]
