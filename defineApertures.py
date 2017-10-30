#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import itertools
import numpy as np
from PIL import Image
from utils import (createBarMask, createBinCircleMask, getDistIma,
                   assignBorderVals)

# %% stimulus settings for the motion-dependent pRF
fovHeight = 11.
pix = 1024
barSize = 1.0
stepSize = 0.5

# %% create ring wedge masks

# derive the radii for the ring limits
minRadi = np.arange(0.5, fovHeight/2.-barSize+stepSize, stepSize)
maxRadi = minRadi + barSize
radiPairs = zip(minRadi, maxRadi)

# derive the angles for the wedge limits
minTheta = np.linspace(0, 360, 8, endpoint=False)
maxTheta = minTheta + 180
thetaPairs = zip(minTheta, maxTheta)

# find all possible combinations between ring and wedge limits
combis = list(itertools.product(radiPairs, thetaPairs))


# %% create masks for the background (no raised cosine)
binMasks = np.empty((pix, pix, len(combis)), dtype='int32')
for ind, combi in enumerate(combis):
    binMasks[..., ind] = createBinCircleMask(fovHeight, pix, rMin=combi[0][0],
                                             rMax=combi[0][1],
                                             thetaMin=combi[1][0],
                                             thetaMax=combi[1][1])
# add frame in the beginning with all zeros
binMasks = np.concatenate((np.zeros((pix, pix)).reshape(pix, pix, 1),
                           binMasks), axis=2)

# for psychopy masks we need numbers in range from -1 to 1 (instead of 0 to 1)
# -1 mean 100 % transperant (not visible) and 1 means 100 % opaque (visible)
opaPgDnMasks = binMasks*2-1

# save as npy array
np.save("/home/marian/Documents/Testing/CircleBarApertures/opa/opaPgDnMasks",
        opaPgDnMasks.astype('int32'))

# save array as images, if wanted
for ind in np.arange(opaPgDnMasks.shape[-1]):
    im = Image.fromarray(opaPgDnMasks[..., ind].astype(np.uint8)*255)
    im.save("/home/marian/Documents/Testing/CircleBarApertures/opa/" +
            "opaPgDnMasks_" + str(ind) + ".png")

# delete opaPgDnMasks to save space
del(opaPgDnMasks)

# %% create masks for the foreground (raised cosine)
opaPgUpMasks = np.empty((pix, pix, binMasks.shape[-1]), dtype='float32')
for i in range(binMasks.shape[-1]):
    # get a single mask
    binMask = binMasks[..., i]
    # check whether there is at least 1 non zero element
    if np.greater(np.sum(binMask), 0):
        # get its distance image
        distIma = getDistIma(binMask, fovHeight, pix)
        # assign raised cosine values to bixels less than 0.5 away from border
        opaPgUpMasks[..., i] = assignBorderVals(binMask, distIma,
                                                borderRange=0.25)
    else:
        # assign old contrast mask
        opaPgUpMasks[..., i] = binMask

# for psychopy masks we need numbers in range from -1 to 1 (instead of 0 to 1)
# -1 mean 100 % transperant (not visible) and 1 means 100 % opaque (visible)
opaPgUpMasks = opaPgUpMasks*2 - 1

# save as npy array
np.save("/home/marian/Documents/Testing/CircleBarApertures/opa/opaPgUpMasks",
        opaPgUpMasks.astype('float32'))

# save array as images, if wanted
for ind in np.arange(opaPgUpMasks.shape[-1]):
    im = Image.fromarray((255*opaPgUpMasks[..., ind]).astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/opa/" +
            "opaPgUpMasks_" + str(ind) + ".png")

# delete opaPgDnMasks to save space
del(opaPgUpMasks)


## %% stimulus settings for the localiser pRF experiment
#fovHeight = 24.
#pix = 1024
#barSize = 3.0
#stepSize = 0.5
#numWedges = 32
#
## derive the x positions for the bar stimulus
#minX = np.arange(0, fovHeight-barSize+stepSize, stepSize)
#maxX = minX + barSize
#xpairs = zip(minX, maxX, np.zeros(len(minX)), np.ones(len(minX))*fovHeight)
#
## derive the y positions for the wedge limits
#minY = np.arange(0, fovHeight-barSize+stepSize, stepSize)
#maxY = minY + barSize
#ypairs = zip(np.zeros(len(minY)), np.ones(len(minY))*fovHeight, minY, maxY)
#
## find all possible combinations between ring and wedge limits
#barCombis = xpairs + ypairs
#
## derive the angles for the wedge stimuli
#minTheta = np.linspace(0, 360, numWedges, endpoint=False)
#maxTheta = minTheta + 45
#thetaPairs = zip(minTheta, maxTheta)
#
#wedgeCombis = zip(np.ones(len(minTheta))*0.3,
#                  np.ones(len(minTheta))*fovHeight/2, minTheta, maxTheta)
#
## %% create bar masks
#barMasks = np.empty((pix, pix, len(barCombis)), dtype='int32')
#for ind, combi in enumerate(barCombis):
#    barMasks[..., ind] = createBarMask(size=fovHeight, numPixel=pix,
#                                       startX=combi[0], stopX=combi[1],
#                                       startY=combi[2], stopY=combi[3])
## add frame in the beginning with all zeros
#barMasks = np.concatenate((np.zeros((pix, pix)).reshape(pix, pix, 1),
#                           barMasks), axis=2)
#
## %% create wedges
#wedgeMasks = np.empty((pix, pix, len(wedgeCombis)), dtype='int32')
#for ind, combi in enumerate(wedgeCombis):
#    wedgeMasks[..., ind] = createBinCircleMask(fovHeight, pix, rMin=combi[0],
#                                               rMax=combi[1],
#                                               thetaMin=combi[2],
#                                               thetaMax=combi[3])
## add frame in the beginning with all zeros
#wedgeMasks = np.concatenate((np.zeros((pix, pix)).reshape(pix, pix, 1),
#                             wedgeMasks), axis=2)
