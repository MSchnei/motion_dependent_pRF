# -*- coding: utf-8 -*-
"""Define stimulus parameters for motion dependent pRF"""

import numpy as np

# %% general stimulus parameters

# number of expected frames per second
nFrames = 60.
# number of pixels for the entire filed of view
pix = 1024.

# height of the entire field of view in deg of vis angle
fovHeight = 17.
# set radius fo the ring/wedge
radius = 5.1
# size of the wedge-ring aperture radial direction in deg of vis angle
barSize = 1.7
# border range for raised cosine ind eg of visual angle
borderRange = 0.6

# how many apertures make one circle?
numAprtCrcle = 6.
# what should be the angle of the ring-wedge
wedgeAngle = 60.

# define opacity on/off cycle in ms
# for stimulus
lenCycStim = 300.
# for raised cosine
lenCycRamp = 50.
# for blank rest
lenCycRest = 600.

# %% set parameters for the texture

# define the spatial frequency of the radial sine wave grating
spatFreq = np.array([1., 2.])
# define the number of angular cycles of the radial sine wave grating
angularCycles = np.ceil(np.pi*fovHeight /
                        ((fovHeight/2.) / ((fovHeight/2)*spatFreq[0]*2))
                        )
# number of cycles per deg of vis angle per second
cycPerSec = np.array([5., 10.])
