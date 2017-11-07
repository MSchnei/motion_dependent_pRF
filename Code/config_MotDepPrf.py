# -*- coding: utf-8 -*-
"""Define stimulus parameters for motion dependent pRF"""

import numpy as np

# %% general stimulus parameters

# number of expected frames per second
nFrames = 60.
# height of the entire field of view in deg of vis angle
fovHeight = 17.
# number of pixels for the entire filed of view
pix = 1024.
# set the minimum radius fo the ring/wedge
minR = 3.4
# size of the bar aperture in deg of vis angle
barSize = 1.7
# by how much will the aperture step through visual field? in deg of vis angle
stepSize = 0.34

# %% set parameters for the texture

# define the spatial frequency of the radial sine wave grating
spatFreq = 1.
# define the number of angular cycles of the radial sine wave grating
angularCycles = np.ceil(np.pi*fovHeight /
                        ((fovHeight/2.) / ((fovHeight/2)*spatFreq*2))
                        )
# number of cycles per deg of vis angle per second
cycPerSec = 5
