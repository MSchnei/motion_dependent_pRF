#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import numpy as np
import itertools

def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    return t, r

def pol2cart(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return(x, y)

def createBinCircleMask(size, numPixel, rMin=0., rMax=500., thetaMin=0.,
                        thetaMax=360.):

    """Create binary wedge-and-ring mask.
    Parameters
    ----------
    size : float
        Size of the (background) square in deg of vis angle
    numPixel : float
        Number of pixels that should be used for the square
    rMin : float
        Minimum radius of the ring apertures in deg of vis angle
    rMax : float
        Maximum radius of the ring apertures in deg of vis angle
    thetaMin : float
        Minimum angle of the wedge apertures in deg
    thetaMax : bool
        Minimum angle of the wedge apertures in deg
    Returns
    -------
    binMask : bool
        binary wedge-and-ring mask
    """    

    # verify that the maximum radius is not bigger than the size 
    if np.greater(rMax, size/2.):
        rMax = np.copy(size/2.)
        print "rMax was reset to max stim size."
    
    # convert from deg to radius
    thetaMin, thetaMax = np.deg2rad((thetaMin, thetaMax))

    # ensure stop angle > start angle
    if thetaMax < thetaMin:
        thetaMax += (2*np.i)

    # create meshgrid
    x, y = np.meshgrid(np.linspace(-size/2., size/2., numPixel),
                       np.linspace(-size/2., size/2., numPixel))

    # convert to polar coordinates
    theta, radius = cart2pol(x, y)
    theta -= thetaMin
    
    # normalize angles so they do not exceed 360 degrees
    theta %= (2*np.pi)

    # define ringMask
    ringMask = np.logical_and(np.greater(radius, rMin),
                              np.less_equal(radius, rMax))
    
    # define wedgeMask
#    wedgeMask = np.logical_and(np.greater(theta, thetaMin),
#                               np.less_equal(theta, thetaMax))
    

    wedgeMask = np.less_equal(theta, thetaMax-thetaMin)

    # return binary mask
    return np.logical_and(ringMask, wedgeMask)


# stimulus settings
fovHeight = 10.
pix = 1200
steps = 20.
barSize = 2.
stepSize = fovHeight/steps

# derive the radii for the ring limits
minRadi = np.linspace(0, fovHeight/2.-stepSize, steps/2.)
maxRadi = minRadi + barSize
radiPairs = zip(minRadi, maxRadi)

# derive the angles for the wedge limits
minTheta = np.linspace(0, 360, 4, endpoint=False)
maxTheta = minTheta + 180
thetaPairs = zip(minTheta, maxTheta)

# find all possible combinations between ring and wedge limits
combis = list(itertools.product(radiPairs, thetaPairs))

# create binary masks
binMasks = np.empty((pix, pix, len(combis)), dtype='bool')
for ind, combi in enumerate(combis):
    binMasks[..., ind] = createBinCircleMask(fovHeight, pix, rMin=combi[0][0],
                                             rMax=combi[0][1],
                                             thetaMin=combi[1][0],
                                             thetaMax=combi[1][1])

