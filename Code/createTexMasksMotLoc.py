#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to creature textures and masks for the motion localiser.
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
from PIL import Image
import config_MotLoc as cfg
from utils import carrierPattern, createBinCircleMask


# %% create the carrier pattern (for the bars)
barTexture = np.zeros((cfg.pix, cfg.pix, cfg.nFrames))
# set the phase of the carrier pattern
phase = np.linspace(0., 4.*np.pi, cfg.nFrames)
lamb = np.sin(phase)/4. + 0.5
for ind, (t, d) in enumerate(zip(phase, lamb)):

    ima = carrierPattern(d, t, cfg.numSquaresBars, cfg.pix)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    barTexture[..., ind] = ima

# save array as images, if wanted
for ind in np.arange(barTexture.shape[2]):
    im = Image.fromarray(255*barTexture[..., ind].astype(np.uint8))
    im.save("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/"
            + "Ima_" + str(ind) + ".png")

# %% create textures (for the bars)
horiBar = np.zeros((barTexture.shape))
vertiBar = np.zeros((barTexture.shape))

horiBar[0:cfg.pix/cfg.numSquaresBars, :, :] = np.copy(
    barTexture[0:cfg.pix/cfg.numSquaresBars, :, :])
vertiBar[:, 0:cfg.pix/cfg.numSquaresBars, :] = np.copy(
    barTexture[:, 0:cfg.pix/cfg.numSquaresBars, :])

# %% create masks (for the bars)

# create templates for horizontal bar
horiBarMask = np.zeros((barTexture.shape[:2] + (cfg.barSteps,)))
horiBarShape = np.zeros((barTexture.shape[:2]))
horiBarShape[0:cfg.pix/cfg.numSquaresBars, :] = 1
# create templates for vertical bar
vertiBarMask = np.zeros((barTexture.shape[:2] + (cfg.barSteps,)))
vertiBarShape = np.zeros((barTexture.shape[:2]))
vertiBarShape[:, 0:cfg.pix/cfg.numSquaresBars] = 1
# create overall circular aperture for all stimuli
circleAperture = createBinCircleMask(cfg.fovHeight, cfg.pix, rLow=cfg.minR,
                                     rUp=cfg.fovHeight/2., thetaMin=0,
                                     thetaMax=360)
# determine the bar positions in pixels
barSizeinPix = cfg.pix*(cfg.barSize/cfg.fovHeight)
posInPix = np.linspace(0, cfg.pix-barSizeinPix, cfg.barSteps)

for ind, pos in enumerate(posInPix):
    # roll forward
    ima = np.roll(horiBarShape, int(pos), axis=0).astype('bool')
    # combine with circle aperture
    ima = np.logical_and(ima, circleAperture)
    # roll back
    ima = np.roll(ima, -int(pos), axis=0).astype('bool')
    horiBarMask[:, :, ind] = ima

for ind, pos in enumerate(posInPix):
    # roll forward
    ima = np.roll(vertiBarShape, int(pos), axis=1).astype('bool')
    # combine with circle aperture
    ima = np.logical_and(ima, circleAperture)
    # roll back
    ima = np.roll(ima, -int(pos), axis=1).astype('bool')
    vertiBarMask[:, :, ind] = ima

horiBarMask = horiBarMask * 2 - 1
vertiBarMask = vertiBarMask * 2 - 1

# %% create the carrier pattern (for the wedge)
wedgeTexture = np.zeros((cfg.pix, cfg.pix, cfg.nFrames))

for ind, (t, d) in enumerate(zip(phase, lamb)):

    ima = carrierPattern(d, t, cfg.numSquaresWedge, cfg.pix)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    wedgeTexture[..., ind] = ima

# %% create the masks for the wedge

# derive the angles for the wedge limits
minTheta = np.linspace(0, 360, cfg.wedgeSteps, endpoint=False)
maxTheta = minTheta + cfg.wedgeWidth
# get combinations of angles and radis
combis = zip(np.zeros(len(minTheta)), np.ones(len(minTheta))*cfg.fovHeight/2.,
             minTheta, maxTheta)
# create the wedgeMasks
wedgeMasks = np.empty((cfg.pix, cfg.pix, len(combis)), dtype='int32')
for ind, combi in enumerate(combis):
    wedgeMasks[..., ind] = createBinCircleMask(cfg.fovHeight, cfg.pix,
                                               rLow=combi[0],
                                               rUp=combi[1],
                                               thetaMin=combi[2],
                                               thetaMax=combi[3],
                                               rMin=cfg.minR)
wedgeMasks = wedgeMasks*2-1

# %%  save textures (for the wedge)
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Textures_MotLoc')

np.savez(filename, horiBar=horiBar, vertiBar=vertiBar, wedge=wedgeTexture)

# %% save masks (for the wedge)
filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Masks_MotLoc')
np.savez(filename, horiBarMask=horiBarMask, vertiBarMask=vertiBarMask,
         wedgeMasks=wedgeMasks)
