# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np

# %% prepare
expectedTR = 2
targetDuration = 0.5
targetDist = targetDuration + 1.2

presOrder = np.hstack((np.zeros(2),
                       np.tile(np.vstack((
                           np.arange(60)[2::15] + 1,
                           np.arange(60)[12::15] + 1)).T.flatten(), 5),
                       np.zeros(1),
                       np.tile(np.vstack((
                           np.arange(60)[2::15] + 1,
                           np.arange(60)[12::15] + 1)).T.flatten(), 5),
                       np.zeros(1),
                       np.tile(np.vstack((
                           np.arange(60)[2::15] + 1,
                           np.arange(60)[12::15] + 1)).T.flatten(), 5),
                       np.zeros(2),
                       ))
apert = np.hstack((np.zeros(2),
                   np.ones(40)*1,
                   np.zeros(1),
                   np.ones(40)*2,
                   np.zeros(1),
                   np.ones(40)*3,
                   np.zeros(2),
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
                        'Conditions_PrepRun_run01')

np.savez(filename, conditions=conditions.astype('int8'), targets=targets,
         targetDuration=targetDuration, targetType=targetType,
         expectedTR=expectedTR)
