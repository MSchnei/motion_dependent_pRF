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
from scipy.stats import gamma

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

# %% prepare presentation order for apetrtures and hyper conditions

# get hyper conditions order, which here means
lstHyperCond = arrangeHyperCondOrder(nrOfCond, nrNullTrialStart,
                                     nrNullTrialBetw, nrNullTrialEnd,
                                     nrOfApertures, cfg.numRep)

# get presentation order of apertures
presOrder = arrangePresOrder(nrOfCond, nrNullTrialStart, nrNullTrialBetw,
                             nrNullTrialEnd, nrOfApertures, cfg.numRep,
                             trialDist)

nrOfHyperCombis = len(lstHyperCond)

# %% concatenating presOrder and hyperCondOrder
lstCnd = []
for ind in np.arange(nrRuns):

    # get hyper condition order
    hyperCondOrder = lstHyperCond[ind % nrOfHyperCombis]
    # create conditions by concatenating presOrder and hyperCondOrder
    conditions = np.vstack((presOrder, hyperCondOrder)).T
    # add to a list
    lstCnd.append(conditions)

condAllRuns = np.vstack(lstCnd)

# %% Remap conditions into space of continuous, unique cond numbers

# arySptCond is remapped into into space of continuous, unique cond numbers
condRmp = np.empty((len(condAllRuns),))

# get the condition nr
condRmp = condAllRuns[:, 0] + condAllRuns[:, 1] * np.max(condAllRuns[:, 0])

# get remapping to continuous numbers
aryFrm = np.unique(condRmp)
aryTo = np.argsort(np.unique(condRmp))

# apply mapping
condRmp = np.array(
    [aryTo[aryFrm == i][0] for i in condRmp])

# %% create neural responses
uniqueCond = np.unique(condRmp)
uniqueCond = uniqueCond[uniqueCond > 0]

nrlTc = np.zeros((len(uniqueCond), len(condRmp)))
for ind, uniCond in enumerate(uniqueCond):
    indPos = np.where(condRmp == uniCond)[0]
    nrlTc[ind, indPos] = 1


# %% convolve with hrf function
def funcHrf(varNumVol, varTr):
    """Create double gamma function.

    Source:
    http://www.jarrodmillman.com/rcsds/lectures/convolution_background.html
    """
    vecX = np.arange(0, varNumVol, 1)

    # Expected time of peak of HRF [s]:
    varHrfPeak = 6.0 / varTr
    # Expected time of undershoot of HRF [s]:
    varHrfUndr = 12.0 / varTr
    # Scaling factor undershoot (relative to peak):
    varSclUndr = 0.35

    # Gamma pdf for the peak
    vecHrfPeak = gamma.pdf(vecX, varHrfPeak)
    # Gamma pdf for the undershoot
    vecHrfUndr = gamma.pdf(vecX, varHrfUndr)
    # Combine them
    vecHrf = vecHrfPeak - varSclUndr * vecHrfUndr

    # Scale maximum of HRF to 1.0:
    vecHrf = np.divide(vecHrf, np.max(vecHrf))

    return vecHrf

# create canonical hrf response
vecHrf = funcHrf(nrlTc.shape[1], expectedTR)

# prepare arrays for convolution by zeropadding
nrlTcCnvl = np.zeros(nrlTc.shape)

vecHrf = np.append(vecHrf, np.zeros(100))
nrlTc = np.concatenate((nrlTc, np.zeros((nrlTc.shape[0], 100))), axis=1)

# Convolve design matrix with HRF model:
for ind, tc in enumerate(nrlTc):
    nrlTcCnvl[ind, :] = np.convolve(tc, vecHrf,
                                    mode='full')[:nrlTc.shape[1]-100]

# %% correlation module
targetCorr = 0.1
corrSwitch = True
while corrSwitch:
    corrMatrix = np.corrcoef(nrlTcCnvl)
    corrMatrixHalf = corrMatrix[np.triu_indices_from(corrMatrix, k=1)]
    corrSwitch = np.invert(np.all(np.less(corrMatrixHalf, targetCorr)))


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


# get target times
targets, targetType = prepareTargets(len(lstHyperCond[0]), expectedTR,
                                     targetDuration, targetDist)



