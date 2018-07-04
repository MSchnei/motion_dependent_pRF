# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import itertools

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

# set number of blank trials
nrNullTrialStart = 7  # 7
nrNullTrialEnd = 7
nrNullTrialBetw = 2

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

# %% Prepare target times
targetSwitch = True
while targetSwitch:
    # prepare targets
    targetTRs = np.zeros(len(conditions)).astype('bool')
    targetPos = np.random.choice(np.arange(3), size=len(conditions),
                                 replace=True,
                                 p=np.array([1/3., 1/3., 1/3.]))
    targetTRs[targetPos == 1] = True
    nrOfTargets = np.sum(targetTRs)

    # prepare random target onset delay
    targetOffsetSec = np.random.uniform(0.1,
                                        expectedTR-targetDuration,
                                        size=nrOfTargets)

    targets = np.arange(0, len(conditions)*expectedTR,
                        expectedTR)[targetTRs]
    targets = targets + targetOffsetSec
    targetSwitch = np.any(np.diff(targets) < targetDist)

# prepare target type
targetType = np.zeros(len(conditions))
targetType[targetTRs] = np.random.choice(np.array([1, 2]),
                                         size=nrOfTargets,
                                         replace=True,
                                         p=np.array([0.5, 0.5]))

# %% save the results

strPathParentUp = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.join(strPathParentUp, 'Conditions',
                        'Conditions_EdgeRun_run08')

np.savez(filename, conditions=conditions.astype('int8'), targets=targets,
         targetDuration=targetDuration, targetType=targetType,
         expectedTR=expectedTR)
