#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:55:45 2017

@author: marian
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
from utils import doRotation


def PrettyPattern(lamb, phase, numSquares, dim):
    """
    Draws a pretty pattern stimulus.

    Parameters:
        lambda
            Wavelength of the sinusoid
        phase
            Phase of the sinusoid
        numSquares
            Number of Squares in the image

    The function returns the new image.
    """

    # 1 square is equivalent to 360 degree input
    width = numSquares * 360
    # what matters is the diagonal, so shrink width with sqrt(2)/2
    width = width * np.sqrt(2)/2

    # get meshgrid of X and Y coordinates
    X, Y = np.meshgrid(np.linspace(-width/2, width/2, dim),
                       np.linspace(-width/2, width/2, dim))
    # rotate by 45 degress
    X, Y = doRotation(X, Y, RotRad=np.pi/4.)
    # set X[0, 0] to 270, since np.sin(270) = -1
    if np.greater_equal(X[0, 0], 0):
        X = X + X[0, 0] + 270
    else:
        X = X - X[0, 0] + 270
    # set Y[0, 0] to 0, since np.cos(0) = 1
    if np.greater_equal(Y[0, 0], 0):
        Y = Y + Y[0, 0]
    else:
        Y = Y - Y[0, 0]
    # Luminance modulation at each pixel
    nom = np.sin(((np.pi*X)/180.)) + np.cos(((np.pi*Y)/180.))
    img = (np.cos(2*np.pi*nom / lamb + phase))

    return img


phase = np.linspace(0., 4.*np.pi, 60.)
lamb = np.sin(phase)/4. + 0.5

dim = 1024
numSquares = 8

noiseTexture = np.zeros((dim, dim, 72))

for ind, (t, d) in enumerate(zip(phase, lamb)):

    ima = PrettyPattern(d, t, numSquares, dim)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    noiseTexture[..., ind] = ima

# save array as images, if wanted
from PIL import Image
for ind in np.arange(noiseTexture.shape[2]):
    im = Image.fromarray(noiseTexture[..., ind].astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
            + "Ima_" + str(ind) + ".png")

# %% create and save textures
horiBar = np.zeros((noiseTexture.shape))
vertiBar = np.zeros((noiseTexture.shape))
littleSquare = np.zeros((noiseTexture.shape))
bigSquare = np.zeros((dim/numSquares, dim/numSquares, noiseTexture.shape[-1]))

horiBar[0:dim/numSquares, :, :] = np.copy(noiseTexture[0:dim/numSquares, :, :])
vertiBar[:, 0:dim/numSquares, :] = np.copy(
    noiseTexture[:, 0:dim/numSquares, :])

littleSquare[0:dim/numSquares, 0:dim/numSquares, :] = np.copy(
    noiseTexture[0:dim/numSquares, 0:dim/numSquares, :])
bigSquare = np.copy(
    noiseTexture[0:dim/numSquares, 0:dim/numSquares, :])

outfile = ("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
           + "textures")
np.savez(outfile, horiBar=horiBar, vertiBar=vertiBar,
         littleSquare=littleSquare, bigSquare=bigSquare)

# %% create and save masks
horiBarMasks = np.zeros((noiseTexture.shape[:2]))
vertiBarMasks = np.zeros((noiseTexture.shape[:2]))
littleSquareMasks = np.zeros((noiseTexture.shape[:2]))
horiBarMasks[0:dim/numSquares, :] = 1
horiBarMasks = horiBarMasks * 2 - 1
vertiBarMasks[:, 0:dim/numSquares] = 1
vertiBarMasks = vertiBarMasks * 2 - 1
littleSquareMasks[0:dim/numSquares, 0:dim/numSquares] = 1
littleSquareMasks = littleSquareMasks * 2 - 1

outfile = ("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
           + "masks")
np.savez(outfile, horiBarMasks=horiBarMasks, vertiBarMasks=vertiBarMasks,
         littleSquareMasks=littleSquareMasks)
