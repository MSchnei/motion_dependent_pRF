#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import itertools
import os
import numpy as np
import config_MotDepPrf as cfg
from PIL import Image
from utils import createBinCircleMask, getDistIma, assignBorderVals, cart2pol

# %% create the radial sine wave pattern

# get cartesian coordinates which are needed to define the textures
x, y = np.meshgrid(np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2., cfg.pix),
                   np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2., cfg.pix))

# get polar coordinates which are needed to define the textures
theta, radius = cart2pol(x, y)
# define the phase for inward/outward conditin
phase = np.linspace(0., 2.*np.pi, cfg.nFrames/cfg.cycPerSec)

# get the array that divides field in angular cycles
polCycles = np.sin(cfg.angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1

# get radial sine wave gratings for main conditions
stimTexture = np.zeros((cfg.pix, cfg.pix, cfg.nFrames/cfg.cycPerSec))
for ind, ph in enumerate(phase):
    ima = np.sin((cfg.fovHeight/2.) * cfg.spatFreq * radius - ph)
    stimTexture[..., ind] = ima

# get radial sine wave gratings for control condition
ctrlTexture = np.zeros((cfg.pix, cfg.pix, 2))
ima = np.sin((cfg.fovHeight/2.) * cfg.spatFreq * radius)
ima = ima * polCycles
ctrlTexture[..., 0] = np.copy(ima)
ctrlTexture[..., 1] = np.copy(ima) * -1

# binarize
stimTexture[np.greater_equal(stimTexture, 0)] = 1
stimTexture[np.less(stimTexture, 0)] = -1
stimTexture = stimTexture.astype('int8')

ctrlTexture[np.greater_equal(ctrlTexture, 0)] = 1
ctrlTexture[np.less(ctrlTexture, 0)] = -1
ctrlTexture = ctrlTexture.astype('int8')


# %%  save textures (for the wedge)
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Textures_MotDepPrf')

np.savez(filename, stimTexture=stimTexture, ctrlTexture=ctrlTexture)

# %% create ring wedge masks

# derive the radii for the ring limits
minRadi = np.arange(0.4+cfg.stepSize-cfg.barSize, cfg.fovHeight/2.,
                    cfg.stepSize)[2:-2]
maxRadi = minRadi + cfg.barSize
radiPairs = zip(minRadi, maxRadi)

# derive the angles for the wedge limits
# add 270 to start at the desired angle
minTheta = np.linspace(0, 360, 6, endpoint=False) + 270
maxTheta = minTheta + 60
thetaPairs = zip(minTheta, maxTheta)

# find all possible combinations between ring and wedge limits
combis = list(itertools.product(radiPairs, thetaPairs))


# %% create masks for the background (no raised cosine)
binMasks = np.empty((cfg.pix, cfg.pix, len(combis)), dtype='int32')
for ind, combi in enumerate(combis):
    binMasks[..., ind] = createBinCircleMask(cfg.fovHeight, cfg.pix,
                                             rLow=combi[0][0],
                                             rUp=combi[0][1],
                                             thetaMin=combi[1][0],
                                             thetaMax=combi[1][1], rMin=0.4,
                                             rMax=5.5)

# %% group masks for the background together (no raised cosine)

# how many aperture make one circle?
numAprtCrcle = 6
jumpInd = np.arange(binMasks.shape[2]).reshape((-1, numAprtCrcle))

# set number of repetitions (number of times the stimulus is shows)
n = 4
lst = list(itertools.product([0, 1], repeat=n))
# flatten the list of tuples and make it a numpy array
for ind, item in enumerate(lst):
    lst[ind] = list(np.hstack(item))
ary = np.array(lst)

# set number of successes (number of times the area should be on)
numSuc = 2
success = np.copy(ary[np.sum(ary, axis=1) == numSuc, :])
# reorder the array
success = np.copy(success[[1, 5, 0, 3, 2, 4], :]).T.astype('bool')

# use index to group apertures together
opaPgDnMasks = np.empty((cfg.pix, cfg.pix, len(combis)/numAprtCrcle*n),
                        dtype='int32')
for ind1, jumpIdx in enumerate(jumpInd):
    for ind2, lgc in enumerate(success):
        # get the right indices
        indices = jumpIdx[lgc]
        # use indices to get relevant apertures
        lgc = binMasks[..., indices.astype('int')]
        opaPgDnMasks[..., ind1*n+ind2] = np.sum(lgc, axis=2).astype('bool')

# add frame in the beginning with all zeros
opaPgDnMasks = np.concatenate((
    np.zeros((cfg.pix, cfg.pix)).reshape(cfg.pix, cfg.pix, 1),
    opaPgDnMasks), axis=2)

# for psychopy masks we need numbers in range from -1 to 1 (instead of 0 to 1)
# -1 mean 100 % transperant (not visible) and 1 means 100 % opaque (visible)
opaPgDnMasks = opaPgDnMasks*2-1

# save array as images, if wanted
for ind in np.arange(opaPgDnMasks.shape[-1]):
    im = Image.fromarray(opaPgDnMasks[..., ind].astype(np.uint8)*255)
    im.save("/home/marian/Documents/Testing/CircleBarApertures/test/" +
            "opaPgDnMasks_" + str(ind) + ".png")

# %% create masks for the foreground (raised cosine)
binMasksRaised = np.empty((cfg.pix, cfg.pix, binMasks.shape[-1]),
                          dtype='float32')
for i in range(binMasks.shape[-1]):
    # get a single mask
    binMask = binMasks[..., i]
    # check whether there is at least 1 non zero element
    if np.greater(np.sum(binMask), 0):
        # get its distance image
        distIma = getDistIma(binMask, cfg.fovHeight, cfg.pix)
        # assign raised cosine values to bixels less than 0.5 away from border
        binMasksRaised[..., i] = assignBorderVals(binMask, distIma,
                                                  borderRange=0.25)
    else:
        # assign old contrast mask
        binMasksRaised[..., i] = binMask

# %% group masks for the foreground together (raised cosine)

# use index to group apertures together
opaPgUpMasks = np.empty((cfg.pix, cfg.pix, len(combis)/numAprtCrcle*n),
                        dtype='float32')
for ind1, jumpIdx in enumerate(jumpInd):
    for ind2, lgc in enumerate(success):
        # get the right indices
        indices = jumpIdx[lgc]
        # use indices to get relevant apertures
        lgc = binMasksRaised[..., indices.astype('int')]
        opaPgUpMasks[..., ind1*n+ind2] = np.sum(lgc, axis=2)

# add frame in the beginning with all zeros
opaPgUpMasks = np.concatenate((
    np.zeros((cfg.pix, cfg.pix)).reshape(cfg.pix, cfg.pix, 1),
    opaPgUpMasks), axis=2)


# for psychopy masks we need numbers in range from -1 to 1 (instead of 0 to 1)
# -1 mean 100 % transperant (not visible) and 1 means 100 % opaque (visible)
opaPgUpMasks = opaPgUpMasks*2 - 1

# save array as images, if wanted
for ind in np.arange(opaPgUpMasks.shape[-1]):
    im = Image.fromarray((255*opaPgUpMasks[..., ind]).astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/test/" +
            "opaPgUpMasks_" + str(ind) + ".png")

# %% save masks as npz array

# restructure arrays such that if indexed with linspace, outward motion results
ouwardInd = np.arange(len(combis)/numAprtCrcle*n).reshape((-1, n)).T
# add 1 to account for thew zero image in the begging
ouwardInd = ouwardInd.flatten() + 1
# add a zero in the beginning for the zero image
ouwardInd = np.hstack((0, ouwardInd))
# reshape the two arrays
opaPgDnMasks = opaPgDnMasks[..., ouwardInd]
opaPgUpMasks = opaPgUpMasks[..., ouwardInd]

# save
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Masks_MotDepPrf')

np.savez(filename, opaPgDnMasks=opaPgDnMasks, opaPgUpMasks=opaPgUpMasks)


# %% older work
## group apertures together to form bow-tie pattern
## set how many apertures should be grouped together
#groupSize = 4
## set the distance between apertures that should be grouped together
#jump = 4
#
## get an index that can be useed as an index
#jumpInd = np.array([])
#for ind in range(jump):
#    jumpInd = np.hstack((jumpInd,
#                         np.arange(ind, binMasks.shape[-1], groupSize*jump)))
#jumpInd = np.sort(jumpInd)
#
#offset = np.array([[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]])
#
#
## use index to group aprtures together
#opaPgDnMasks = np.empty((cfg.pix, cfg.pix, len(jumpInd)), dtype='int32')
#for idx, jumpIdx in enumerate(jumpInd):
#    # get indices
#    indices = np.linspace(jumpIdx, jumpIdx+jump*(groupSize-1), groupSize)
#
#    # add random offset
#    indices = indices + offset[idx % 4, :]
#
#    # use indices to geth relevant apertures
#    lgc = binMasks[..., indices.astype('int')]
#
#    opaPgDnMasks[..., idx] = np.sum(lgc, axis=2).astype('bool')
#
## add frame in the beginning with all zeros
#opaPgDnMasks = np.concatenate((np.zeros((cfg.pix, cfg.pix)).reshape(cfg.pix, cfg.pix, 1),
#                               opaPgDnMasks), axis=2)
#
## for psychopy masks we need numbers in range from -1 to 1 (instead of 0 to 1)
## -1 mean 100 % transperant (not visible) and 1 means 100 % opaque (visible)
#opaPgDnMasks = opaPgDnMasks*2-1
#
## save as npy array
#np.save("/home/marian/Documents/Testing/CircleBarApertures/test/opaPgDnMasks",
#        opaPgDnMasks.astype('int32'))
#
## save array as images, if wanted
#for ind in np.arange(opaPgDnMasks.shape[-1]):
#    im = Image.fromarray(opaPgDnMasks[..., ind].astype(np.uint8)*255)
#    im.save("/home/marian/Documents/Testing/CircleBarApertures/test/" +
#            "opaPgDnMasks_" + str(ind) + ".png")