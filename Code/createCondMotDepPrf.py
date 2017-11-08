# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import itertools
import os

# %% set paramters
nrOfApertures = 60

expectedTR = 2
targetDuration = 0.3

# total number of conditions
nrOfCond = 3

# set number of blank trials
nrNullTrialStart = 1  # 12
nrNullTrialEnd = 12
nrNullTrialBetw = 12

# %% prepare

# prepare vectors for presentation order
barApertures = np.linspace(1, nrOfApertures, nrOfApertures)

# prepare modules of presentation order
presOrder1 = np.hstack((np.zeros(nrNullTrialStart),
                        barApertures,
                        np.zeros(nrNullTrialBetw),
                        barApertures,
                        np.zeros(nrNullTrialBetw),
                        barApertures,
                        np.zeros(nrNullTrialEnd),
                        ))

presOrder2 = np.hstack((np.zeros(nrNullTrialStart),
                        barApertures[::-1],
                        np.zeros(nrNullTrialBetw),
                        barApertures[::-1],
                        np.zeros(nrNullTrialBetw),
                        barApertures[::-1],
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
                          np.ones(nrOfApertures)*indCond[0],
                          np.zeros(nrNullTrialBetw),
                          np.ones(nrOfApertures)*indCond[1],
                          np.zeros(nrNullTrialBetw),
                          np.ones(nrOfApertures)*indCond[2],
                          np.zeros(nrNullTrialEnd),
                          ))

    # concatenate presOrder and aperture
    conditions1 = np.vstack((presOrder1, aperture)).T
    conditions2 = np.vstack((presOrder2, aperture)).T

    # %% Prepare target times

    # prepare targets
    nrOfTargets = int(len(conditions1)/6)
    targetTRs = np.zeros(len(conditions1))
    lgcRep = True
    while lgcRep:
        targetPos = np.random.choice(np.arange(nrNullTrialStart,
                                     len(conditions1)-nrNullTrialEnd),
                                     nrOfTargets,
                                     replace=False)
        lgcRep = np.greater(np.sum(np.diff(np.sort(targetPos)) == 1), 0)
    targetTRs[targetPos] = 1
    assert nrOfTargets == np.sum(targetTRs)

    # prepare random target onset delay
    targetOffsetSec = np.random.uniform(0.1,
                                        expectedTR-targetDuration,
                                        size=nrOfTargets)

    # %% save the results

    strPathParentUp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if 2*ind+1 > 9:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run' + str(2*ind+1))
    else:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run0' + str(2*ind+1))
    np.savez(filename1, conditions=conditions1,
             targetTRs=targetTRs, targetOffsetSec=targetOffsetSec,
             targetDuration=targetDuration, expectedTR=expectedTR)
    if 2*ind+2 > 9:
        filename2 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run' + str(2*ind+2))
    else:
        filename2 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_MotDepPrf_run0' + str(2*ind+2))
    np.savez(filename2, conditions=conditions2,
             targetTRs=targetTRs, targetOffsetSec=targetOffsetSec,
             targetDuration=targetDuration, expectedTR=expectedTR)
