#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to creature textures and masks for the motion localiser.
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import itertools
from utils import carrierPattern, createBinCircleMask

# %% set parameters

# set the height of the field of view in deg
fovHeight = 24.
# set the number of frames
nFrames = 60.
# set the dimension (numbe rof pixels for the carrier pattern)
dim = 1024

# set the phase of the carrier pattern
phase = np.linspace(0., 4.*np.pi, nFrames)
lamb = np.sin(phase)/4. + 0.5

# set the number of squares that the carrier pattern should be made up of
# for the bars
numSquaresBars = 8
# set the number of squares that the carrier pattern should be made up of
# for the wedge
numSquaresWedge = 8

# set the number of steps that the wedge should take
wedgeSteps = 32
# set the width of the width in deg
wedgeWidth = 45
# set the minimum radius fo the wedge
minR = 0.4

# %% create the carrier pattern (for the bars)
barTexture = np.zeros((dim, dim, nFrames))

for ind, (t, d) in enumerate(zip(phase, lamb)):

    ima = carrierPattern(d, t, numSquaresBars, dim)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    barTexture[..., ind] = ima

# save array as images, if wanted
from PIL import Image
for ind in np.arange(barTexture.shape[2]):
    im = Image.fromarray(barTexture[..., ind].astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
            + "Ima_" + str(ind) + ".png")

# %% create textures (for the bars)
horiBar = np.zeros((barTexture.shape))
vertiBar = np.zeros((barTexture.shape))

horiBar[0:dim/numSquaresBars, :, :] = np.copy(
    barTexture[0:dim/numSquaresBars, :, :])
vertiBar[:, 0:dim/numSquaresBars, :] = np.copy(
    barTexture[:, 0:dim/numSquaresBars, :])

# %% create masks (for the bars)
horiBarMask = np.zeros((barTexture.shape[:2]))
vertiBarMask = np.zeros((barTexture.shape[:2]))
horiBarMask[0:dim/numSquaresBars, :] = 1
horiBarMask = horiBarMask * 2 - 1
vertiBarMask[:, 0:dim/numSquaresBars] = 1
vertiBarMask = vertiBarMask * 2 - 1

# %% create the carrier pattern (for the wedge)
wedgeTexture = np.zeros((dim, dim, nFrames))

for ind, (t, d) in enumerate(zip(phase, lamb)):

    ima = carrierPattern(d, t, numSquaresWedge, dim)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    wedgeTexture[..., ind] = ima

# %% create the masks for the wedge

# derive the angles for the wedge limits
minTheta = np.linspace(0, 360, wedgeSteps, endpoint=False)
maxTheta = minTheta + wedgeWidth
# get combinations of angles and radis
combis = zip(np.zeros(len(minTheta)), np.ones(len(minTheta))*fovHeight/2.,
             minTheta, maxTheta)
# create the wedgeMasks
wedgeMasks = np.empty((dim, dim, len(combis)), dtype='int32')
for ind, combi in enumerate(combis):
    wedgeMasks[..., ind] = createBinCircleMask(fovHeight, dim,
                                               rLow=combi[0],
                                               rUp=combi[1],
                                               thetaMin=combi[2],
                                               thetaMax=combi[3],
                                               rMin=minR)
wedgeMasks = wedgeMasks*2-1
# %%  save textures (for the wedge)
outfile = ("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
           + "textures")
np.savez(outfile, horiBar=horiBar, vertiBar=vertiBar, wedge=wedgeTexture)

# %% save masks (for the wedge)
outfile = ("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
           + "masks")
np.savez(outfile, horiBarMask=horiBarMask, vertiBarMask=vertiBarMask,
         wedgeMasks=wedgeMasks)
