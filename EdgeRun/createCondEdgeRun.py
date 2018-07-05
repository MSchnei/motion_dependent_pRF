# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import itertools
from utils import funcHrf, prepareTargets

# %% set paramters
expectedTR = 2
targetDuration = 0.5
targetDist = targetDuration + 1.2

# total number of motion conditions
nrOfMtnCond = 3
# total number of eccentricity conditions
nrOfEccCond = 3
# total number of repetitions
nrReps = 5
# total number of runs
nrRuns = 6

# set number of blank trials
nrNullTrialStart = 7  # 7
nrNullTrialEnd = 7
nrNullTrialBetw = 2

# set number of attempts over which the script should try to minimize the
# correlation between conditions
numAtt = 10000

# %%

eccCond = np.linspace(1, nrOfEccCond, nrOfEccCond)
motionCond = np.linspace(1, nrOfMtnCond, nrOfMtnCond)

# get all possible combination
combi = list(itertools.product(eccCond, motionCond))
# transform to numpy array
repBlock = np.empty((2,))
for ind, item in enumerate(combi):
    repBlock = np.vstack((repBlock, np.hstack(item)))
repBlock = repBlock[1:, :]

# %% try to minimize the cxorrelation between conditions

# prepare variable that can store the temporary correlation value
varCorrMaxWnr = 1.0

for att in np.arange(numAtt):

    lstRuns = []
    for indRun in np.arange(nrRuns):

        # %%
        # scale to number of repretitions
        conditions = np.repeat(repBlock, nrReps, axis=0)

        # shuffle
        np.random.shuffle(conditions)
        #  insert blank trials inbetween the conditions
        conditions = np.insert(conditions,
                               np.repeat(np.arange(conditions.shape[0]),
                                         nrNullTrialBetw),
                               np.zeros((nrNullTrialBetw,)),
                               axis=0)
        # insert blank trials in the beginning and end
        conditions = np.vstack((np.zeros((nrNullTrialStart, 2)),
                                conditions,
                                np.zeros((nrNullTrialStart, 2)),
                                ))
        lstRuns.append(conditions)

    conditions = np.vstack(lstRuns)

    # %% Remap conditions into space of continuous, unique cond numbers

    # arySptCond is remapped into into space of continuous, unique cond numbers
    condRmp = np.empty((len(conditions),))

    # get the condition nr
    condRmp = conditions[:, 0] + conditions[:, 1] * np.max(conditions[:, 0])

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

    # demean
    nrlTcCnvl = np.subtract(nrlTcCnvl, np.mean(nrlTcCnvl, axis=1)[:, None])
    # standardize
    nrlTcCnvl = np.divide(nrlTcCnvl, np.std(nrlTcCnvl, axis=1)[:, None])

    # %% correlation module

    corrMatrix = np.corrcoef(nrlTcCnvl)
    corrMatrixHalf = corrMatrix[np.triu_indices_from(corrMatrix, k=1)]
    tempMax = np.max(np.abs(corrMatrixHalf))
    tempMean = np.mean(np.abs(corrMatrixHalf))
    print("---attempt " + str(att))
    print("------max " + str(tempMax))
    print("------mean " + str(tempMean))

    if np.less(tempMax, varCorrMaxWnr):
        varCorrMaxWnr = np.copy(tempMax)
        varCorrMeanWnr = np.copy(tempMean)
        conditionsWnr = np.copy(lstRuns)
        tempVals = np.copy(corrMatrixHalf)

# %% save the results

for ind in np.arange(nrRuns):

    # retrieve conditions for every run
    conditions = conditionsWnr[ind, ...]

    # get target times and types
    targets, targetType = prepareTargets(len(conditions), expectedTR,
                                         targetDuration, targetDist)

    strPathParentUp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if ind+1 > 9:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_EdgeRun_run' + str(ind+1))
    else:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_EdgeRun_run0' + str(ind+1))
    np.savez(filename1, conditions=conditions.astype('int8'), targets=targets,
             targetDuration=targetDuration, targetType=targetType,
             expectedTR=expectedTR)
