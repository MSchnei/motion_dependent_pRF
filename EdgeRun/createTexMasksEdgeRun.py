#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import itertools
import os
import numpy as np
import config_EdgeRun as cfg
from PIL import Image
from utils import createBinCircleMask, getDistIma, assignBorderVals, cart2pol

# %% create the radial sine wave pattern
# get cartesian coordinates which are needed to define the textures
x, y = np.meshgrid(np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.),
                   np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.))

# get polar coordinates which are needed to define the textures
theta, radius = cart2pol(x, y)
# define the phase for inward/outward conditin
phase = np.linspace(0., 2.*np.pi, cfg.nFrames/cfg.cycPerSec)

# get the array that divides field in angular cycles
polCycles = np.sin(cfg.angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1

# get radial sine wave gratings for main conditions
stimTexture = np.zeros((cfg.pix/2., cfg.pix/2., cfg.nFrames/cfg.cycPerSec))
for ind, ph in enumerate(phase):
    ima = np.sin(cfg.spatFreq * 2. * np.pi * radius - ph)
    stimTexture[..., ind] = ima

# get radial sine wave gratings for control condition
ctrlTexture = np.zeros((cfg.pix/2., cfg.pix/2., 2))
ima = np.sin(cfg.spatFreq * 2. * np.pi * radius)
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
                        'Textures_EdgeRun')

np.savez(filename, stimTexture=stimTexture.astype('int8'),
         ctrlTexture=ctrlTexture.astype('int8'))

# %% create ring wedge masks

# derive the radii for the ring limits
minRadi = np.linspace(cfg.minR, cfg.fovHeight/2.,
                      (cfg.fovHeight/2. - cfg.minR) / cfg.barSize,
                      endpoint=False)
maxRadi = minRadi + cfg.barSize
radiPairs = zip(minRadi, maxRadi)

# derive the angles for the wedge limits
# add 270 to start at the desired angle
minTheta = np.linspace(0, 360, cfg.numAprtCrcle, endpoint=False) + 270
maxTheta = minTheta + cfg.wedgeAngle
thetaPairs = zip(minTheta, maxTheta)

# find all possible combinations between ring and wedge limits
combis = list(itertools.product(radiPairs, thetaPairs))


# %% create masks for the background (no raised cosine)
binMasks = np.empty((cfg.pix, cfg.pix, len(combis)), dtype='int8')
for ind, combi in enumerate(combis):
    binMasks[..., ind] = createBinCircleMask(cfg.fovHeight, cfg.pix,
                                             rLow=combi[0][0],
                                             rUp=combi[0][1],
                                             thetaMin=combi[1][0],
                                             thetaMax=combi[1][1],
                                             rMax=cfg.fovHeight/2.)

# %% group masks for the background together (no raised cosine)

# get an array that indexes the right masks
lst = [[1, 3, 5], [7, 9, 11], [13, 15, 17]]

# use index to group apertures together
opaPgDnMasks = np.empty((cfg.pix, cfg.pix, len(lst)), dtype='int8')
for ind1 in np.arange(len(lst)):
        # get the right indices
        indices = lst[ind1]
        # use indices to get relevant apertures
        lgc = binMasks[..., indices]
        opaPgDnMasks[..., ind1] = np.sum(lgc, axis=2).astype('bool')

# add frame in the beginning with all zeros
opaPgDnMasks = np.concatenate((
    np.zeros((cfg.pix, cfg.pix)).reshape(cfg.pix, cfg.pix, 1),
    opaPgDnMasks), axis=2)


# %% create masks for the foreground (raised cosine)

opaPgUpMasks = np.empty((cfg.pix, cfg.pix, opaPgDnMasks.shape[-1]),
                        dtype='float32')
for i in range(opaPgDnMasks.shape[-1]):
    # get a single mask
    binMask = opaPgDnMasks[..., i]
    # check whether there is at least one element of 1
    if np.any(binMask == 1):
        print "image " + str(i)
        # get its distance image
        distIma = getDistIma(binMask, cfg.fovHeight, cfg.pix)
        # assign raised cosine values to bixels less than 0.5 away from border
        opaPgUpMasks[..., i] = assignBorderVals(binMask, distIma,
                                                borderRange=cfg.borderRange)
    else:
        # assign old contrast mask
        opaPgUpMasks[..., i] = binMask


# %% save masks as npz array

# for psychopy masks we need numbers in range from -1 to 1 (instead of 0 to 1)
# -1 mean 100 % transperant (not visible) and 1 means 100 % opaque (visible)
opaPgDnMasks = opaPgDnMasks*2 - 1
opaPgUpMasks = opaPgUpMasks*2 - 1

# save
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Masks_EdgeRun')

np.savez(filename, opaPgDnMasks=opaPgDnMasks.astype('int8'),
         opaPgUpMasks=opaPgUpMasks.astype('float32'))
