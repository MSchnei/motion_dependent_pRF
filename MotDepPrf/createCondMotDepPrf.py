# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
import config_MotDepPrf as cfg
from utils import (arrangePresOrder, arrangeHyperCondOrder, prepareTargets,
                   funcHrf)

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
nrNullTrialReps = 2

# set trial distance
trialDist = 2

# set number of attempts over which the script should try to minimize the
# correlation between conditions
numAtt = 100

# %% try to minimize the cxorrelation between conditions

# prepare variable that can store the temporary correlation value
varCorrTmpWnr = 1.0

# get hyper conditions order, which here means different oders of
# flicker, outward and inward motion
lstHyperCond = arrangeHyperCondOrder(nrOfCond, nrNullTrialStart,
                                     nrNullTrialBetw, nrNullTrialEnd,
                                     nrNullTrialReps, nrOfApertures,
                                     cfg.numRep)

# get the number of possible combinations of all the hyper conditions
nrOfHyperCombis = len(lstHyperCond)

for att in np.arange(numAtt):

    # %% concatenating presOrder and hyperCondOrder
    lstCnd = []
    for ind in np.arange(nrRuns):

        # %% get presentation order of apertures
        presOrder = arrangePresOrder(nrOfCond, nrNullTrialStart,
                                     nrNullTrialBetw, nrNullTrialEnd,
                                     nrNullTrialReps, nrOfApertures,
                                     cfg.numRep, trialDist)

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

    corrMatrix = np.corrcoef(nrlTcCnvl)
    corrMatrixHalf = corrMatrix[np.triu_indices_from(corrMatrix, k=1)]
    tempMax = np.max(np.abs(corrMatrixHalf))
    print("---attempt " + str(att))
    print("------max " + str(tempMax))
    print("------mean " + str(np.mean(np.abs(corrMatrixHalf))))

    if np.less(tempMax, varCorrTmpWnr):
        varCorrTmpWnr = np.copy(tempMax)
        presOrderTmpWnr = np.copy(lstCnd)
        tempVals = np.copy(corrMatrixHalf)

# %% save the results

for ind in np.arange(nrRuns):

    # retrieve conditions from winner array
    conditions = presOrderTmpWnr[ind, ...]

    # get target times and types
    targets, targetType = prepareTargets(len(lstHyperCond[0]), expectedTR,
                                         targetDuration, targetDist)

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
