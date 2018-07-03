# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
import config_MotDepPrf as cfg
from utils import arrangePresOrder, arrangeHyperCondOrder, prepareTargets

# %% set paramters

# set number of desired runs
nrRuns = 12

# derive number of apertures
nrOfApertures = len(np.arange(cfg.minR+cfg.stepSize-cfg.barSize,
                              cfg.fovHeight/2., cfg.stepSize)[2:-2])

expectedTR = 2.
targetDuration = 0.5
targetDist = targetDuration + 1.2

# total number of conditions
nrOfCond = 3

# set number of blank trials
nrNullTrialStart = 7  # 7
nrNullTrialEnd = 7
nrNullTrialBetw = 14

# set trial distance
trialDist = 2

# %% prepare presentation order for hyper conditions and targets

# get hyper conditions order, which here means
lstHyperCond = arrangeHyperCondOrder(nrOfCond, nrNullTrialStart,
                                     nrNullTrialBetw, nrNullTrialEnd,
                                     nrOfApertures, cfg.numRep)

# get target times
targets, targetType = prepareTargets(len(lstHyperCond[0]), expectedTR,
                                     targetDuration, targetDist)

# %%
# get presentation order of apertures
presOrder = arrangePresOrder(nrOfCond, nrNullTrialStart, nrNullTrialBetw,
                             nrNullTrialEnd, nrOfApertures, cfg.numRep,
                             trialDist)



# %% save the results

nrOfHyperCombis = len(lstHyperCond)

for ind in np.arange(nrRuns):

    # get hyper condition order
    hyperCondOrder = lstHyperCond[ind % nrOfHyperCombis]
    # create conditions by concatenating presOrder and hyperCondOrder
    conditions = np.vstack((presOrder, hyperCondOrder)).T

    strPathParentUp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if ind+1 > 9:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run' + str(ind+1))
    else:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run0' + str(ind+1))
    np.savez(filename1, conditions=conditions.astype('int8'), targets=targets,
             targetDuration=targetDuration, targetType=targetType,
             expectedTR=expectedTR)

# %% correlation module
targetCorr = 0.1
corrSwitch = True
while corrSwitch:
    timeCourses = np.random.random((3*60, 222*12))
    corrMatrix = np.corrcoef(timeCourses)
    corrMatrixHalf = corrMatrix[np.triu_indices_from(corrMatrix, k=1)]
    corrSwitch = np.invert(np.all(np.less(corrMatrixHalf, targetCorr)))
