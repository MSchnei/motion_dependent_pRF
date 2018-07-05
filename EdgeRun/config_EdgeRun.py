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
# set the minimum radius fo the ring/wedge
minR = 3.4
# size of the wedge-ring aperture radial direction in deg of vis angle
barSize = 1.7
# by how much will the aperture step through visual field? in deg of vis angle
stepSize = 0.34
# border range for raised cosine ind eg of visual angle
borderRange = 0.6
# how many apertures make one circle?
numAprtCrcle = 6
# what should be the angle of the ring-wedge
wedgeAngle = 60.
# set number of repetitions (number of times the stimulus is shown)
numRep = 4

# define opacity on/off cycle in ms
# for stimulus
lenCycStim = 300.
# for raised cosine
lenCycRamp = 50.
# for blank rest
lenCycRest = 600.

# %% set parameters for the texture

# define the spatial frequency of the radial sine wave grating
spatFreq = 1.
# define the number of angular cycles of the radial sine wave grating
angularCycles = np.ceil(np.pi*fovHeight /
                        ((fovHeight/2.) / ((fovHeight/2)*spatFreq*2))
                        )
# number of cycles per deg of vis angle per second
cycPerSec = 5.
