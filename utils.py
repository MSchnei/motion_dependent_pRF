# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 18:53:04 2017

@author: marian
"""
import numpy as np
from scipy import spatial, signal


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
    return t, r


def pol2cart(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return(x, y)


def doRotation(x, y, RotRad=0):
    """Generate a meshgrid and rotate it by RotRad radians."""

    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                          [-np.sin(RotRad), np.cos(RotRad)]])

    rot = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))
    rot = np.transpose(rot, (1, 2, 0))
    return rot[..., 0], rot[..., 1]


def createBarMask(size=24, numPixel=1024, startX=0, stopX=24, startY=0,
                  stopY=24):
    """Create binary bar mask.
    Parameters
    ----------
    size : float
        Size of the (background) square in deg of vis angle
    numPixel : float
        Number of pixels that should be used for the square
    startX : float
        Minimum x of bar in deg of vis angle
    stopX : float
        Maximum x of bar in deg of vis angle
    startY : float
        Minimum y of bar in deg of vis angle
    stopY : bool
        Maximum y of bar in deg of vis angle
    Returns
    -------
    binMask : bool
        binary bar mask
    """
    # create meshgrid
    x, y = np.meshgrid(np.linspace(0, size, numPixel, endpoint=False),
                       np.linspace(0, size, numPixel, endpoint=False))
    xcontrs = np.logical_and(np.greater_equal(x, startX), np.less(x, stopX))
    yconstr = np.logical_and(np.greater_equal(y, startY), np.less(y, stopY))
    return np.logical_and(xcontrs, yconstr)


def createBinCircleMask(size, numPixel, rLow=0., rUp=500., thetaMin=0.,
                        thetaMax=360., rMin=0, rMax=np.inf):

    """Create binary wedge-and-ring mask.
    Parameters
    ----------
    size : float
        Size of the (background) square in deg of vis angle
    numPixel : float
        Number of pixels that should be used for the square
    rLow : float
        Lower radius of the ring apertures in deg of vis angle
    rUp : float
        Upper radius of the ring apertures in deg of vis angle
    thetaMin : float
        Minimum angle of the wedge apertures in deg
    thetaMax : bool
        Minimum angle of the wedge apertures in deg
    thetaMin : float
        Minimum radius of the ring apertures in deg of vis angle
    thetaMax : bool
        Maximum radius of the ring apertures in deg of vis angle
    Returns
    -------
    binMask : bool
        binary wedge-and-ring mask
    """

    # verify that the upper radius is not bigger than the max radius
    if np.greater(rUp, rMax):
        rUp = np.copy(rMax)
        print "rUp was reset to max stim size."

    # verify that the lower radius is not less than the min radius
    if np.less(rLow, rMin):
        rLow = np.copy(rMin)
        print "rLow was reset to min stim size."

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
    ringMask = np.logical_and(np.greater(radius, rLow),
                              np.less_equal(radius, rUp))

    wedgeMask = np.less_equal(theta, thetaMax-thetaMin)

    # return binary mask
    return np.logical_and(ringMask, wedgeMask)


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
    window = signal.hann(100)[:50]
    # get new values
    newvals = np.copy(window[distance.astype('int')])
    # take the origanal mask, the logical and insert new values
    binMask = binMask.astype('float')
    binMask[lgcIma] = np.copy(newvals)

    return binMask

def time2frame(t, frameRate=60):
    """Convert time to frames"""
    # time wil be between 0 and TR
    # frames should be between 0 and TR*frameRate
    return t*frameRate


# create function to time ramped onsets and offsets
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
