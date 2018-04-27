# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:30:25 2017

@author: marian
"""

import numpy as np
from PIL import Image
import os

# %% set parameters
inPath = "/home/marian/Documents/Git/motion_dependent_pRF"
outPath = "/media/sf_D_DRIVE/MotDepPrf/Analysis/S02/02_MotLoc/prfPngs/"

# provide name of motLoc files in the order that they were shown
lstMotLoc = [
    "Conditions_MotLoc_run02.npz",
    "Conditions_MotLoc_run03.npz",
    "Conditions_MotLoc_run04.npz",
    "Conditions_MotLoc_run01.npz",
    ]

factorX = 8
factorY = 8

# value to multipy mask value (1s) with for png format
scaleValue = 255

# %% get masks
# Path of mask files:
strPthMsk = (inPath
             + os.path.sep
             + 'MaskTextures'
             + os.path.sep
             + 'Masks_MotLoc.npz')

# Load npz file content:
with np.load((strPthMsk)) as objMsks:
    horiBarMasksFitting = objMsks["horiBarMasksFitting"]
    vertiBarMasksFitting = objMsks["vertiBarMasksFitting"]
    wedgeMasksFitting = objMsks["wedgeMasksFitting"]

# %% get conditions
lstCond = []
# Path of condition file
strPthCond = (inPath
              + os.path.sep
              + 'Conditions'
              + os.path.sep)

# Loop through npz files in target directory:
for cond in lstMotLoc:
    inputFile = os.path.join(strPthCond, cond)
    # extract condition
    npzfile = np.load(inputFile)
    conditions = npzfile["conditions"]
    # append condition to list
    lstCond.append(conditions)

# join conditions across runs
conditions = np.vstack(lstCond)

# load np arrays from dictionary and save their 2D slices as png
for index in np.arange(conditions.shape[0]):
    # get the index of the masks and conditions
    keyMask = conditions[index, 0]
    keyCond = conditions[index, 1]

    if keyCond == 0:
        ima = np.zeros(wedgeMasksFitting.shape[:2])
    elif keyCond == 1:
        ima = horiBarMasksFitting[..., keyMask-1]
    elif keyCond == 2:
        ima = vertiBarMasksFitting[..., keyMask-1]
    elif keyCond == 3:
        ima = wedgeMasksFitting[..., keyMask-1]

    # if desired, downsample
    if factorX > 1 or factorY > 1:
        ima = ima[::factorX, ::factorY]

    im = Image.fromarray(scaleValue * ima.astype(np.uint8))
    if index > 999:
        filename = ("frame" + '' + str(index) + '.png')
    elif index > 99:
        filename = ("frame" + '0' + str(index) + '.png')
    elif index > 9:
        filename = ("frame" + '00' + str(index) + '.png')
    else:
        filename = ("frame" + '000' + str(index) + '.png')

    im.save((os.path.join(outPath, filename)))
