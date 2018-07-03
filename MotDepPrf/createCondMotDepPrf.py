# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import itertools
import numpy as np
import config_MotDepPrf as cfg


# %% set paramters

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

# %% prepare presentation order for different apertures

# create differences that are all larger than 5 (to avoid overlap)
trialSwitch = True
while trialSwitch:
    # prepare presentation order
    trials = np.tile(np.linspace(1, nrOfApertures, nrOfApertures),
                     (cfg.numRep, 1))
    # shuffle every row independetly
    idx = np.argsort(np.random.random(trials.shape), axis=1)
    trials = trials[np.arange(trials.shape[0])[:, None], idx]
    # check whether any difference between neighbors is smaller than trialDist
    # if so, continue the loop, if not interrupt
    trialSwitch = np.invert(np.all(np.all(np.abs(np.diff(trials)) >= trialDist,
                                          axis=1)))

# turn trial numbers into unique condition identifiers
trials = trials + (nrOfApertures * np.arange(cfg.numRep))[:, None]
barApertures = trials.flatten()


# %% prepare vectors for presentation order

# prepare modules of presentation order
presOrder = np.hstack((np.zeros(nrNullTrialStart),
                       barApertures,
                       np.zeros(nrNullTrialBetw),
                       barApertures,
                       np.zeros(nrNullTrialBetw),
                       barApertures,
                       np.zeros(nrNullTrialEnd),
                       ))


# prepare to indicate weather flicker, expanding or contracting motion
# counterbalance occurance of
lst = list(itertools.permutations(np.arange(nrOfCond)+1, nrOfCond))
for ind, item in enumerate(lst):
    lst[ind] = list(np.hstack(item))
aryPerm = np.array(lst)

# loop through combinations
for ind, indCond in enumerate(aryPerm):
    aperture = np.hstack((np.zeros(nrNullTrialStart),
                          np.ones(nrOfApertures*cfg.numRep)*indCond[0],
                          np.zeros(nrNullTrialBetw),
                          np.ones(nrOfApertures*cfg.numRep)*indCond[1],
                          np.zeros(nrNullTrialBetw),
                          np.ones(nrOfApertures*cfg.numRep)*indCond[2],
                          np.zeros(nrNullTrialEnd),
                          ))

    # concatenate presOrder and aperture
    conditions = np.vstack((presOrder, aperture)).T

    # %% Prepare target times
    targetSwitch = True
    while targetSwitch:
        # prepare targets
        targetTRs = np.zeros(len(conditions1)).astype('bool')
        targetPos = np.random.choice(np.arange(3), size=len(conditions1),
                                     replace=True,
                                     p=np.array([1/3., 1/3., 1/3.]))
        targetTRs[targetPos == 1] = True
        nrOfTargets = np.sum(targetTRs)

        # prepare random target onset delay
        targetOffsetSec = np.random.uniform(0.1,
                                            expectedTR-targetDuration,
                                            size=nrOfTargets)

        targets = np.arange(0, len(conditions1)*expectedTR,
                            expectedTR)[targetTRs]
        targets = targets + targetOffsetSec
        targetSwitch = np.any(np.diff(targets) < targetDist)

    # prepare target type
    targetType = np.zeros(len(conditions1))
    targetType[targetTRs] = np.random.choice(np.array([1, 2]),
                                             size=nrOfTargets,
                                             replace=True,
                                             p=np.array([0.5, 0.5]))

    # %% save the results

    strPathParentUp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if 2*ind+1 > 9:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run' + str(2*ind+1))
    else:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run0' + str(2*ind+1))
    np.savez(filename1, conditions=conditions1.astype('int8'), targets=targets,
             targetDuration=targetDuration, targetType=targetType,
             expectedTR=expectedTR)
    if 2*ind+2 > 9:
        filename2 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run' + str(2*ind+2))
    else:
        filename2 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run0' + str(2*ind+2))
    np.savez(filename2, conditions=conditions2.astype('int8'), targets=targets,
             targetDuration=targetDuration, targetType=targetType,
             expectedTR=expectedTR)
