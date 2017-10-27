#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import itertools
import numpy as np
from scipy import spatial


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
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

    wedgeMask = np.less_equal(theta, thetaMax-thetaMin)

    # return binary mask
    return np.logical_and(ringMask, wedgeMask)


def raisedCos(steps, T=0.5, beta=0.5):
    """"Create binary wedge-and-ring mask.
    Parameters
    ----------
    steps : float
        Number of points in the output window
    T: float
        The symbol-period
    beta : float
        Roll-off factor
    Returns
    -------
    hf : 1d np.array
        Raised-cosine filter in frequency space
    """

    frequencies = np.linspace(-1/T, 1/T, steps)
    hf = np.empty(len(frequencies))
    for ind, f in enumerate(frequencies):
        if np.less_equal(np.abs(f), (1-beta)/(2*T)):
            hf[ind] = 1
        elif np.logical_and(np.less_equal(np.abs(f), (1+beta)/(2*T)),
                            np.greater(np.abs(f), (1-beta)/(2*T))):
            hf[ind] = 0.5*(1+np.cos((np.pi*T/2)*(np.abs(f)-(1-beta)/2*T)))
        else:
            hf[ind] = 0
    return hf


def getDistIma(inputIma, fovHeight=10, pix=512):
    # create meshgrid
    x, y = np.meshgrid(np.linspace(-fovHeight/2., fovHeight/2., pix),
                       np.linspace(-fovHeight/2., fovHeight/2., pix))
    # identify border voxels
    grad = np.gradient(inputIma)
    gramag = np.greater(np.sqrt(np.power(grad[0], 2) + np.power(grad[1], 2)),
                        0)
    border = np.logical_and(gramag, inputIma)
    # get (degree) coordinates for points on the border
    borderpoints = np.vstack((x[border], y[border])).T
    # get (degree) coordinates for all points in the image
    allpoints = np.vstack((x.flatten(), y.flatten())).T
    # get distace of all points in the image to every border voxel
    distance = spatial.distance.cdist(allpoints, borderpoints,
                                      metric='euclidean')
    # get the distance to the border point that is closest
    distMin = np.min(distance, axis=1)
    # put minimum distances in images shape.
    distIma = distMin.reshape((pix, pix))

    return distIma


def assignBorderVals(binMask, distIma, borderRange=0.5):
    "Assign the new (raised cosine values) to voxels in desired border range."

    # find logical for pixels away less than a certain number from border
    lgcIma = np.logical_and(np.less_equal(distIma, borderRange), binMask)
    # get distances for values that fullfill logical
    distance = distIma[lgcIma]
    # distance will contain values from 0 to border range
    scaleFactor = 50 / borderRange
    # scale distances to fit in the window
    distance *= scaleFactor
    # get raised cosine window
    window = raisedCos(100, T=1, beta=0.9)[:50]
    # get new values
    newvals = np.copy(window[distance.astype('int')])
    # take the origanal mask, the logical and insert new values
    binMask = binMask.astype('float')
    binMask[lgcIma] = np.copy(newvals)

    return binMask


# stimulus settings
fovHeight = 10.
pix = 512
steps = 20.
barSize = 1.5
stepSize = fovHeight/steps

# derive the radii for the ring limits
minRadi = np.linspace(0, fovHeight/2.-stepSize, steps/2.)
maxRadi = minRadi + barSize
radiPairs = zip(minRadi, maxRadi)

# derive the angles for the wedge limits
minTheta = np.linspace(0, 360, 8, endpoint=False)
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

binMasks = np.concatenate((np.zeros((pix, pix)).reshape(pix, pix, 1),
                           binMasks), axis=2)
np.save("/home/marian/Documents/Testing/CircleBarApertures/Masks", binMasks)
# save array as images, if wanted
from PIL import Image
for ind in np.arange(binMasks.shape[2]):
    im = Image.fromarray(binMasks[..., ind].astype(np.uint8)*255)
    im.save("/home/marian/Documents/Testing/CircleBarApertures/Ima" + "_" +
            str(ind) + ".png")

# create masks with raised cosines
binMasksRamped = np.empty((pix, pix, binMasks.shape[-1]), dtype='bool')

for i in range(binMasks.shape[-1]):
    # get a single mask
    binMask = binMasks[..., i]
    # check whether there is at least 1 non zero element
    if np.greater(np.sum(binMask), 0):
        # get its distance image
        distIma = getDistIma(binMask, fovHeight, pix)
        # assign raised cosine values to bixels less than 0.5 away from border
        binMasksRamped[..., i] = assignBorderVals(binMask, distIma,
                                                  borderRange=0.5)
    else:
        # assign old mask
        binMasksRamped[..., i] = binMask


np.save("/home/marian/Documents/Testing/CircleBarApertures/ramped/RampedMasks",
        binMasksRamped)
# save array as images, if wanted
from PIL import Image
for ind in np.arange(binMasks.shape[2]):
    im = Image.fromarray(binMasks[..., ind].astype(np.uint8)*255)
    im.save("/home/marian/Documents/Testing/CircleBarApertures/ramped/Ima" +
            "_" + str(ind) + ".png")


## get a single mask
#binMask = binMasks[..., 20]
## get its distance image
#distIma = getDistIma(binMask, fovHeight, pix)
## assign new, raised cosine values to bixels less than 0.5 away from border
#binMask = assignBorderVals(binMask, distIma, borderRange=0.5)
