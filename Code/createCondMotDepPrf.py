# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:21:45 2016

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os

# %% set paramters
NrOfApertures = 60

ExpectedTR = 2
TargetDuration = 0.3

# total number of conditions
NrOfCond = 3

# set number of blank trials
NrNullTrialStart = 1  # 12
NrNullTrialEnd = 12
NrNullTrialBetw = 12


# prepare vector for motion direction; define 8 directions of motion
BarApertures = np.linspace(1, NrOfApertures, NrOfApertures)


# prepare vector for presentation order
presOrder1 = np.hstack((np.zeros(NrNullTrialStart),
                        BarApertures,
                        np.zeros(NrNullTrialBetw),
                        BarApertures,
                        np.zeros(NrNullTrialBetw),
                        BarApertures,
                        np.zeros(NrNullTrialEnd),
                        ))

presOrder2 = np.hstack((np.zeros(NrNullTrialStart),
                        BarApertures[::-1],
                        np.zeros(NrNullTrialBetw),
                        BarApertures[::-1],
                        np.zeros(NrNullTrialBetw),
                        BarApertures[::-1],
                        np.zeros(NrNullTrialEnd),
                        ))

# prepare to indicate weather horizontal, vertical bar apeture or wedge
aperture = np.hstack((np.zeros(NrNullTrialStart),
                      np.ones(len(BarApertures))*1,
                      np.zeros(NrNullTrialBetw),
                      np.ones(len(BarApertures))*2,
                      np.zeros(NrNullTrialBetw),
                      np.ones(len(BarApertures))*3,
                      np.zeros(NrNullTrialEnd),
                      ))

# concatenate presOrder and aperture
Conditions1 = np.vstack((presOrder1, aperture)).T
Conditions2 = np.vstack((presOrder2, aperture)).T


# %% Prepare target times

# prepare targets
NrOfTargets = int(len(Conditions1)/6)
Targets = np.zeros(len(Conditions1))
lgcRep = True
while lgcRep:
    TargetPos = np.random.choice(np.arange(NrNullTrialStart,
                                 len(Conditions1)-NrNullTrialEnd), NrOfTargets,
                                 replace=False)
    lgcRep = np.greater(np.sum(np.diff(np.sort(TargetPos)) == 1), 0)
Targets[TargetPos] = 1
assert NrOfTargets == np.sum(Targets)

# prepare random target onset delay
TargetOnsetinSec = np.random.uniform(0.1,
                                     ExpectedTR-TargetDuration,
                                     size=NrOfTargets)

# %% save the results

str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
filename1 = os.path.join(str_path_parent_up, 'Conditions',
                         'Conditions_MotDepPrf_run01')
np.savez(filename1, Conditions1=Conditions1,
         TargetTRs=Targets, TargetOnsetinSec=TargetOnsetinSec,
         TargetDuration=TargetDuration, ExpectedTR=ExpectedTR)

filename2 = os.path.join(str_path_parent_up, 'Conditions',
                         'Conditions_MotDepPrf_run02')
np.savez(filename2, Conditions2=Conditions2,
         TargetTRs=Targets, TargetOnsetinSec=TargetOnsetinSec,
         TargetDuration=TargetDuration, ExpectedTR=ExpectedTR)