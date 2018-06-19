#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:47:54 2018

@author: marian
"""

import numpy as np
from scipy import spatial, mgrid


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
    return t, r


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


def getDistIma(inputIma, fovHeight=10, pix=512, pixSlct=None, borderType=None,
               normalize=False):
    # create meshgrid
    x, y = np.meshgrid(np.linspace(-fovHeight/2., fovHeight/2., pix),
                       np.linspace(-fovHeight/2., fovHeight/2., pix))
    # identify all border pixels
    if borderType is None:
        grad = np.gradient(inputIma)
        gramag = np.greater(np.sqrt(np.power(grad[0], 2) +
                                    np.power(grad[1], 2)),
                            0)
        border = np.logical_and(gramag, inputIma)
    # only identify pixels at the lower border of the stimulus aperture
    elif borderType == "lower":
        grad = np.gradient(inputIma)[0]
        border = np.less(grad, 0.0)
    # only identify pixels at the upper border of the stimulus aperture
    elif borderType == "upper":
        grad = np.gradient(inputIma)[0]
        border = np.greater(grad, 0.0)
    # only identify pixels at the right border of the stimulus aperture
    elif borderType == "right":
        grad = np.gradient(inputIma)[1]
        border = np.less(grad, 0.0)
    # only identify pixels at the left border of the stimulus aperture
    elif borderType == "left":
        grad = np.gradient(inputIma)[1]
        border = np.greater(grad, 0.0)

    # get (degree) coordinates for points on the border
    x, y = np.meshgrid(np.linspace(-fovHeight/2., fovHeight/2., pix),
                       np.linspace(-fovHeight/2., fovHeight/2., pix))
    borderpoints = np.vstack((x[border], y[border])).T

    # since input imae will be sued to index from now on,
    # we make sure that it is in boolean
    inputIma = inputIma.astype(np.bool)

    # get (degree) coordinates for all points in image or in stimulus aperture
    if pixSlct is None:
        # get coordinates for all points in the image
        points = np.vstack((x.flatten(), y.flatten())).T
    elif pixSlct == "stimAprt":
        # only get coordinates for points in stimulus aperture
        points = np.vstack((x[inputIma],
                            y[inputIma])).T

    # get distace of all points in the image to every border voxel
    distance = spatial.distance.cdist(points, borderpoints,
                                      metric='euclidean')
    # get the distance to the border point that is closest
    distMin = np.min(distance, axis=1)
    # put minimum distances in images shape
    pixCrdX, pixCrdY = np.meshgrid(np.linspace(0, pix, pix, endpoint=False,
                                               dtype=np.int32),
                                   np.linspace(0, pix, pix, endpoint=False,
                                               dtype=np.int32))

    # put minimum distances in images shape.
    if pixSlct is None:
        distIma = distMin.reshape((pix, pix))
    elif pixSlct == "stimAprt":
        distIma = np.zeros((int(pix), int(pix)), dtype=np.float32)
        distIma[inputIma] = distMin

    # normalize the image to mean 1, if desired
    if normalize:
        distIma[inputIma] = np.subtract(distIma[inputIma],
                                        np.mean(distIma[inputIma]))
        distIma[inputIma] += 1

    return distIma


def crt_2D_gauss(varSizeX, varSizeY, varPosX, varPosY, varSd):
    """Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSd : float, positive
        Standard deviation of 2D Gauss.
    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    Reference
    ---------
    [1] mathworld.wolfram.com/GaussianFunction.html
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # create x and y in meshgrid:
    aryX, aryY = mgrid[0:varSizeX, 0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        (np.square((aryX - varPosX)) + np.square((aryY - varPosY))) /
        (2.0 * np.square(varSd))
        )
    aryGauss = np.exp(-aryGauss) / (2 * np.pi * np.square(varSd))

    return aryGauss


def crt_2D_ani_gauss(varSizeX, varSizeY, varPosX, varPosY, varSdX, varSdY):
    """Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSdX : float, positive
        Standard deviation of 2D Gauss in x direction.
    varSdX : float, positive
        Standard deviation of 2D Gauss in y direction.
    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    Reference
    ---------
    [1] mathworld.wolfram.com/GaussianFunction.html
    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # create x and y in meshgrid:
    aryX, aryY = mgrid[0:varSizeX, 0:varSizeY]

    # The actual creation of the Gaussian array:
    aryGauss = (
        np.square((aryX - varPosX)) / (2.0 * np.square(varSdX)) +
        np.square((aryY - varPosY)) / (2.0 * np.square(varSdY))
        )
    aryGauss = np.exp(-aryGauss) / (2 * np.pi * varSdX * varSdY)

    return aryGauss


def crt_2D_rot_gauss(varSizeX, varSizeY, varPosX, varPosY, varSdX, varSdY,
                     varTh):
    """Create 2D Gaussian kernel.

    Parameters
    ----------
    varSizeX : int, positive
        Width of the visual field.
    varSizeY : int, positive
        Height of the visual field..
    varPosX : int, positive
        X position of centre of 2D Gauss.
    varPosY : int, positive
        Y position of centre of 2D Gauss.
    varSdX : float, positive
        Standard deviation of 2D Gauss in x direction.
    varSdX : float, positive
        Standard deviation of 2D Gauss in y direction.
    Returns
    -------
    aryGauss : 2d numpy array, shape [varSizeX, varSizeY]
        2d Gaussian.
    Reference
    ---------
    [1] mathworld.wolfram.com/GaussianFunction.html

    """
    varSizeX = int(varSizeX)
    varSizeY = int(varSizeY)

    # create x and y in meshgrid:
    aryX, aryY = mgrid[0:varSizeX, 0:varSizeY]

    # calculate a, b, c
    a = (np.square(np.cos(varTh)) / (2.0 * np.square(varSdX)) +
         np.square(np.sin(varTh)) / (2.0 * np.square(varSdY)))
    b = (np.sin(2*varTh) / (4.0 * np.square(varSdX)) +
         np.sin(2*varTh) / (4.0 * np.square(varSdY)))
    c = (np.square(np.sin(varTh)) / (2.0 * np.square(varSdX)) +
         np.square(np.cos(varTh)) / (2.0 * np.square(varSdY)))

    # The actual creation of the Gaussian array:
    aryGauss = np.exp(-(a * np.square(aryX - varPosX) +
                        2 * b * (aryX - varPosX) * (aryY - varPosY) +
                        c * np.square(aryY - varPosY)))
    return aryGauss


def rmp_rng(aryVls, varNewMin, varNewMax, varOldThrMin=None,
            varOldAbsMax=None):
    """Remap values in an array from

    Parameters
    ----------
    aryVls : 1D numpy array
        Array with values that need to be remapped.
    varNewMin : float
        Desired minimum value of new, remapped array.
    varNewMax : float
        Desired maximum value of new, remapped array.
    varOldThrMin : float
        Theoretical minimum of old distribution. Can be specified if this
        theoretical minimum does not occur in empirical distribution but
        should be considered nontheless.
    varOldThrMin : float
        Theoretical maximum of old distribution. Can be specified if this
        theoretical maximum does not occur in empirical distribution but
        should be considered nontheless.

    Returns
    -------
    aryVls : 1D numpy array
        Array with remapped values.
    """
    if varOldThrMin is None:
        varOldMin = aryVls.min()
    else:
        varOldMin = varOldThrMin
    if varOldAbsMax is None:
        varOldMax = aryVls.max()
    else:
        varOldMax = varOldAbsMax

    aryNewVls = np.empty((aryVls.shape), dtype=aryVls.dtype)
    for ind, val in enumerate(aryVls):
        aryNewVls[ind] = (((val - varOldMin) * (varNewMax - varNewMin)) /
                          (varOldMax - varOldMin)) + varNewMin

    return aryNewVls


def rmp_deg_pixel_x_y_s(vecX, vecY, vecPrfSd, tplPngSize,
                        varExtXmin, varExtXmax, varExtYmin, varExtYmax):
    """Remap x, y, sigma parameters from degrees to pixel.

    Parameters
    ----------
    vecX : 1D numpy array
        Array with possible x parametrs in degree
    vecY : 1D numpy array
        Array with possible y parametrs in degree
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in degree
    tplPngSize : tuple, 2
        Pixel dimensions of the visual space in pixel (width, height).
    varExtXmin : float, negative or 0
        Extent of visual space from centre in negative x-direction (width)
    varExtXmax : float, positive or 0
        Extent of visual space from centre in positive x-direction (width)
    varExtYmin : int, negative or 0
        Extent of visual space from centre in negative y-direction (height)
    varExtYmax : float, positive or 0
        Extent of visual space from centre in positive y-direction (height)
    Returns
    -------
    vecX : 1D numpy array
        Array with possible x parametrs in pixel
    vecY : 1D numpy array
        Array with possible y parametrs in pixel
    vecPrfSd : 1D numpy array
        Array with possible sd parametrs in pixel
    """
    # Remap modelled x-positions of the pRFs:
    vecXpxl = rmp_rng(vecX, 0.0, (tplPngSize[0] - 1), varOldThrMin=varExtXmin,
                      varOldAbsMax=varExtXmax)

    # Remap modelled y-positions of the pRFs:
    vecYpxl = rmp_rng(vecY, 0.0, (tplPngSize[1] - 1), varOldThrMin=varExtYmin,
                      varOldAbsMax=varExtYmax)

    # We calculate the scaling factor from degrees of visual angle to
    # pixels separately for the x- and the y-directions (the two should
    # be the same).
    varDgr2PixX = tplPngSize[0] / (varExtXmax - varExtXmin)
    varDgr2PixY = tplPngSize[1] / (varExtYmax - varExtYmin)

    # Check whether varDgr2PixX and varDgr2PixY are similar:
    strErrMsg = 'ERROR. The ratio of X and Y dimensions in ' + \
        'stimulus space (in degrees of visual angle) and the ' + \
        'ratio of X and Y dimensions in the upsampled visual space' + \
        'do not agree'
    assert 0.5 > np.absolute((varDgr2PixX - varDgr2PixY)), strErrMsg

    # Convert prf sizes from degrees of visual angles to pixel
    vecPrfSdpxl = np.multiply(vecPrfSd, varDgr2PixX)

    return vecXpxl, vecYpxl, vecPrfSdpxl


def cnvl_2D_gauss(aryGauss, arySptExpInf, tplPngSize):
    """Spatially convolve input with 2D Gaussian model.

    Parameters
    ----------
    aryGauss : 2d numpy array, shape [tplPngSize]
        Array with the prf model.
    arySptExpInf : 3d numpy array, shape [n_x_pix, n_y_pix, n_conditions]
        All spatial conditions stacked along second axis.
    tplPngSize : tuple, 2.
        Pixel dimensions of the visual space (width, height).
    Returns
    -------
    data : 2d numpy array, shape [n_models, n_conditions]
        Closed data.
    Reference
    ---------
    [1]
    """
    if aryGauss.ndim == 3:
        # Multiply pixel-time courses with Gaussian pRF models:
        aryCndTcTmp = np.multiply(arySptExpInf, aryGauss)
    elif aryGauss.ndim == 2:
        # Multiply pixel-time courses with Gaussian pRF models:
        aryCndTcTmp = np.multiply(arySptExpInf, aryGauss[:, :, None])

    # Calculate sum across x- and y-dimensions - the 'area under the
    # Gaussian surface'.
    aryCndTcTmp = np.sum(aryCndTcTmp, axis=(0, 1))

    return aryCndTcTmp


def cnvl_grad_prf(prf_old, grad, Normalize=True):

    prf_new = prf_old * grad

    if Normalize:
        varOldSum = np.sum(prf_old, axis=(0, 1))
        prf_new = np.divide(prf_new, np.sum(prf_new, axis=(0, 1))) * varOldSum

    return prf_new
