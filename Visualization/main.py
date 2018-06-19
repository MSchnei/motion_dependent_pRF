#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:57:13 2018

@author: marian
"""

import os
import itertools
import numpy as np
from utils import (createBinCircleMask, getDistIma, cnvl_grad_prf, cart2pol,
                   crt_2D_gauss, cnvl_2D_gauss, rmp_deg_pixel_x_y_s)
import matplotlib.pyplot as plt

# %% Set parameters

# number of pixels for the entire filed of view
pix = 1024.
# height of the entire field of view in deg of vis angle
fovHeight = 17.
# set the minimum radius fo the ring/wedge
minR = 3.4
# size of the wedge-ring aperture radial direction in deg of vis angle
barSize = 1.7
# by how much will the aperture step through visual field? in deg of vis angle
stepSize = 0.34
# border range for raised cosine ind eg of visual angle
borderRange = 0.6
# how many apertures make one circle?
numAprtCrcle = 1.
# what should be the angle of the ring-wedge
wedgeAngle = 180.


# %% Create stimulus apertures

# derive the radii for the ring limits
minRadi = np.arange(minR+stepSize-barSize, fovHeight/2., stepSize)[2:-2]
maxRadi = minRadi + barSize
radiPairs = zip(minRadi, maxRadi)

# derive the angles for the wedge limits
# add 270 to start at the desired angle
minTheta = np.linspace(0, 360, numAprtCrcle, endpoint=False) + 270
maxTheta = minTheta + wedgeAngle
thetaPairs = zip(minTheta, maxTheta)

# find all possible combinations between ring and wedge limits
combis = list(itertools.product(radiPairs, thetaPairs))

# %% create masks for the background (no raised cosine)

aryImaNtr = np.empty((pix, pix, len(combis)), dtype='int8')
for ind, combi in enumerate(combis):
    aryImaNtr[..., ind] = createBinCircleMask(fovHeight, pix,
                                              rLow=combi[0][0],
                                              rUp=combi[0][1],
                                              thetaMin=combi[1][0],
                                              thetaMax=combi[1][1],
                                              rMin=minR,
                                              rMax=fovHeight/2.)

# %% add gradients as a function of distance to border

aryImaOtw = np.zeros((aryImaNtr.shape))
aryImaInw = np.zeros((aryImaNtr.shape))

for indIma in range(aryImaNtr.shape[-1]):
    # get image
    inputIma = aryImaNtr[..., indIma]

    # add gradients as a function of distance to border for outward motion
    aryImaOtw[..., indIma] = getDistIma(inputIma, fovHeight=fovHeight, pix=pix,
                                        pixSlct="stimAprt", borderType="right",
                                        normalize=True)
    # add gradients as a function of distance to border for inward motion
    aryImaInw[..., indIma] = getDistIma(inputIma, fovHeight=fovHeight, pix=pix,
                                        pixSlct="stimAprt", borderType="left",
                                        normalize=True)

for indIma in range(aryImaNtr.shape[-1]):
    # get image
    inputIma = aryImaNtr[..., indIma]
    varFigName = os.path.join("/media/sf_D_DRIVE/MotDepPrf/Presentation",
                              "figures", "Figure_prfshift",
                              "stim_" + str(indIma) + ".svg")
    plt.imsave(varFigName, inputIma, cmap='viridis')


# %% Create 2d Gaussian model
vecX = np.array([5.95])
vecY = np.array([0])
vecSd = np.array([0.5])

mdlPrms = rmp_deg_pixel_x_y_s(vecY, vecX, vecSd, (int(pix), int(pix)),
                              -fovHeight/2., fovHeight/2.,
                              -fovHeight/2., fovHeight/2.)
mdlPrms = np.stack(mdlPrms, axis=1)

prf_ntr = crt_2D_gauss(pix, pix, mdlPrms[:, 0], mdlPrms[:, 1], mdlPrms[:, 2])

# %% apply gradient tp prf
gridX, gridY = np.meshgrid(np.linspace(-pix/2., pix/2., pix, endpoint=False),
                           np.linspace(-pix/2., pix/2., pix, endpoint=False))

_, gridRad = cart2pol(gridX, gridY)

gradOtw = np.copy(np.ones(gridRad.shape)*gridRad.max() - gridRad)
gradInw = np.copy(gridRad)

gradOtw = np.divide(gradOtw, np.sum(gradOtw, axis=(0, 1)))
gradInw = np.divide(gradInw, np.sum(gradInw, axis=(0, 1)))

prf_otw = cnvl_grad_prf(prf_ntr, gradOtw, Normalize=True)
prf_inw = cnvl_grad_prf(prf_ntr, gradInw, Normalize=True)

# %% save new prfs to disk
fig, ax = plt.subplots()
cax = plt.imshow(prf_otw, cmap="viridis")
varFigName = os.path.join("/media/sf_D_DRIVE/MotDepPrf/Presentation/figures",
                          "Figure_prfshift", "prf_otw" + ".svg")

cbar = fig.colorbar(cax,
                    ticks=[prf_otw.min(), prf_otw.max()/2., prf_otw.max()],
                    orientation='horizontal')
cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar

fig.savefig(varFigName,
            dpi=400,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            papertype='a0',
            transparent=False,
            frameon=None,
            bbox_inches="tight",
            )

# %% calculate neural responses

nrlRspNtr = cnvl_2D_gauss(prf_ntr, aryImaNtr, (pix, pix)).T
nrlRspOtw = cnvl_2D_gauss(prf_otw, aryImaNtr, (pix, pix)).T
nrlRspInw = cnvl_2D_gauss(prf_inw, aryImaNtr, (pix, pix)).T

# define x positions
x = minRadi + barSize/2.
print("simulated prf responses")
fig, ax = plt.subplots()
ax.set_color_cycle(['#d95f02', '#7570b3', '#1b9e77'])
plt.plot(x, nrlRspNtr, x, nrlRspOtw, x, nrlRspInw, linewidth=3)
plt.legend(['neutral', 'outward', 'inward'], fontsize=16,
           loc='center left', bbox_to_anchor=(1, 0.5))

varFigName = os.path.join("/media/sf_D_DRIVE/MotDepPrf/Presentation/figures",
                          "Figure_prfshift", "neural_responses" + ".svg")
fig.savefig(varFigName,
            dpi=400,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            papertype='a0',
            transparent=False,
            frameon=None,
            bbox_inches="tight",
            )
