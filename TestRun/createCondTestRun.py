# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
from utils import balancedLatinSquares


# %% set paramters
expectedTR = 2
targetDuration = 0.5
targetDist = targetDuration + 1.2

# total number of conditions
nrOfCond = 3

# set number of blank trials
nrNullTrialStart = 7  # 7
nrNullTrialEnd = 7
nrNullTrialBetw = 14

# prepare vectors for presentation order
apertures = np.linspace(1, 5, 5)

# counterbalance occurance of vertiBar, horiBar and wedge
# add 1 since the wedge appears twice
lst = balancedLatinSquares(len(apertures))
for ind, item in enumerate(lst):
    lst[ind] = list(np.hstack(item))
aryPerm = np.array(lst)

presOrder = np.hstack((np.zeros(nrNullTrialStart),
                       np.hstack((aryPerm)),
                       np.zeros(nrNullTrialBetw),
                       np.hstack((aryPerm)),
                       np.zeros(nrNullTrialBetw),
                       np.hstack((aryPerm)),
                       np.zeros(nrNullTrialEnd),
                       ))

apert = np.hstack((np.zeros(nrNullTrialStart),
                   np.ones(len(aryPerm.flatten()))*1,
                   np.zeros(nrNullTrialBetw),
                   np.ones(len(aryPerm.flatten()))*2,
                   np.zeros(nrNullTrialBetw),
                   np.ones(len(aryPerm.flatten()))*3,
                   np.zeros(nrNullTrialEnd),
                   ))

conditions = np.vstack((presOrder, apert)).T


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
                        'Conditions_TestRun_run01')

np.savez(filename, conditions=conditions.astype('int8'), targets=targets,
         targetDuration=targetDuration, targetType=targetType,
         expectedTR=expectedTR)
