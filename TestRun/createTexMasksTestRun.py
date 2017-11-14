#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import itertools
import os
import numpy as np
import config_TestRun as cfg
from utils import createBinCircleMask, getDistIma, assignBorderVals, cart2pol

# %% create the radial sine wave pattern for cond 1, 2, 3 (regular texture)

# get cartesian coordinates which are needed to define the textures
x, y = np.meshgrid(np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.),
                   np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.))

# get polar coordinates which are needed to define the textures
theta, radius = cart2pol(x, y)
# define the phase for inward/outward conditin
phase = np.linspace(0., 2.*np.pi, cfg.nFrames/cfg.cycPerSec[0])

# get the array that divides field in angular cycles
polCycles = np.sin(cfg.angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1

# get radial sine wave gratings for main conditions
stimTexture1 = np.zeros((cfg.pix/2., cfg.pix/2., cfg.nFrames/cfg.cycPerSec[0]))
for ind, ph in enumerate(phase):
    ima = np.sin(cfg.spatFreq[0] * 2. * np.pi * radius - ph)
    stimTexture1[..., ind] = ima

# get radial sine wave gratings for control condition
ctrlTexture1 = np.zeros((cfg.pix/2., cfg.pix/2., 2))
ima = np.sin(cfg.spatFreq[0] * 2. * np.pi * radius)
ima = ima * polCycles
ctrlTexture1[..., 0] = np.copy(ima)
ctrlTexture1[..., 1] = np.copy(ima) * -1

# binarize
stimTexture1[np.greater_equal(stimTexture1, 0)] = 1
stimTexture1[np.less(stimTexture1, 0)] = -1
stimTexture1 = stimTexture1.astype('int8')

ctrlTexture1[np.greater_equal(ctrlTexture1, 0)] = 1
ctrlTexture1[np.less(ctrlTexture1, 0)] = -1
ctrlTexture1 = ctrlTexture1.astype('int8')

# %% create the radial sine wave pattern for cond 4 (double speed)

# get cartesian coordinates which are needed to define the textures
x, y = np.meshgrid(np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.),
                   np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.))

# get polar coordinates which are needed to define the textures
theta, radius = cart2pol(x, y)
# define the phase for inward/outward conditin
phase = np.linspace(0., 2.*np.pi, cfg.nFrames/cfg.cycPerSec[1])

# get the array that divides field in angular cycles
polCycles = np.sin(cfg.angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1

# get radial sine wave gratings for main conditions
stimTexture2 = np.zeros((cfg.pix/2., cfg.pix/2., cfg.nFrames/cfg.cycPerSec[1]))
for ind, ph in enumerate(phase):
    ima = np.sin(cfg.spatFreq[0] * 2. * np.pi * radius - ph)
    stimTexture2[..., ind] = ima

# get radial sine wave gratings for control condition
ctrlTexture2 = np.zeros((cfg.pix/2., cfg.pix/2., 2))
ima = np.sin(cfg.spatFreq[0] * 2. * np.pi * radius)
ima = ima * polCycles
ctrlTexture2[..., 0] = np.copy(ima)
ctrlTexture2[..., 1] = np.copy(ima) * -1

# binarize
stimTexture2[np.greater_equal(stimTexture2, 0)] = 1
stimTexture2[np.less(stimTexture2, 0)] = -1
stimTexture2 = stimTexture2.astype('int8')

ctrlTexture2[np.greater_equal(ctrlTexture2, 0)] = 1
ctrlTexture2[np.less(ctrlTexture2, 0)] = -1
ctrlTexture2 = ctrlTexture2.astype('int8')

# %% create the radial sine wave pattern for cond 5 (double frequency)

# get cartesian coordinates which are needed to define the textures
x, y = np.meshgrid(np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.),
                   np.linspace(-cfg.fovHeight/2., cfg.fovHeight/2.,
                               cfg.pix/2.))

# get polar coordinates which are needed to define the textures
theta, radius = cart2pol(x, y)
# define the phase for inward/outward conditin
phase = np.linspace(0., 2.*np.pi, cfg.nFrames/cfg.cycPerSec[0])

# get the array that divides field in angular cycles
polCycles = np.sin(cfg.angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1

# get radial sine wave gratings for main conditions
stimTexture3 = np.zeros((cfg.pix/2., cfg.pix/2., cfg.nFrames/cfg.cycPerSec[0]))
for ind, ph in enumerate(phase):
    ima = np.sin(cfg.spatFreq[1] * 2. * np.pi * radius - ph)
    stimTexture3[..., ind] = ima

# get radial sine wave gratings for control condition
ctrlTexture3 = np.zeros((cfg.pix/2., cfg.pix/2., 2))
ima = np.sin(cfg.spatFreq[1] * 2. * np.pi * radius)
ima = ima * polCycles
ctrlTexture3[..., 0] = np.copy(ima)
ctrlTexture3[..., 1] = np.copy(ima) * -1

# binarize
stimTexture3[np.greater_equal(stimTexture3, 0)] = 1
stimTexture3[np.less(stimTexture3, 0)] = -1
stimTexture3 = stimTexture3.astype('int8')

ctrlTexture3[np.greater_equal(ctrlTexture3, 0)] = 1
ctrlTexture3[np.less(ctrlTexture3, 0)] = -1
ctrlTexture3 = ctrlTexture3.astype('int8')


# %%  save textures (for the wedge)
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Textures_TestRun')

np.savez(filename, stimTexture1=stimTexture1.astype('int8'),
         stimTexture2=stimTexture2.astype('int8'),
         stimTexture3=stimTexture3.astype('int8'),
         ctrlTexture1=ctrlTexture1.astype('int8'),
         ctrlTexture2=ctrlTexture2.astype('int8'),
         ctrlTexture3=ctrlTexture3.astype('int8'))

# %% create ring wedge masks for cond

# derive the radii for the ring limits
minRadi = cfg.radius
maxRadi = minRadi + cfg.barSize
radiPairs = [(minRadi, maxRadi)]

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
lst = [[1, 3, 5],
       [0, 2, 4],
       [0, 1, 2, 3, 4, 5]]

# use index to group apertures together
opaPgDnMasks = np.empty((cfg.pix, cfg.pix, len(lst)), dtype='int8')
for ind1 in np.arange(3):
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
                        'Masks_TestRun')

np.savez(filename, opaPgDnMasks=opaPgDnMasks.astype('int8'),
         opaPgUpMasks=opaPgUpMasks.astype('float32'))
