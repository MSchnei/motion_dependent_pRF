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
NrOfBarApertures = 43
NrOfWedgeApertures = 32

ExpectedTR = 2
TargetDuration = 0.3

# total number of conditions
NrOfCond = 3

# set number of blank trials
NrNullTrialStart = 1  # 10
NrNullTrialEnd = 10
NrNullTrialBetw = 20


# prepare vector for motion direction; define 8 directions of motion
BarApertures = np.linspace(1, NrOfBarApertures, NrOfBarApertures)
WedgeApertures = np.linspace(1, NrOfWedgeApertures, NrOfWedgeApertures)


# prepare vector for presentation order
presOrder1 = np.hstack((np.zeros(NrNullTrialStart),
                        BarApertures, BarApertures[::-1],
                        np.zeros(NrNullTrialBetw),
                        BarApertures, BarApertures[::-1],
                        np.zeros(NrNullTrialBetw),
                        WedgeApertures, WedgeApertures[::-1],
                        np.zeros(NrNullTrialEnd),
                        ))

presOrder2 = np.hstack((np.zeros(NrNullTrialStart),
                        BarApertures[::-1], BarApertures,
                        np.zeros(NrNullTrialBetw),
                        BarApertures[::-1], BarApertures,
                        np.zeros(NrNullTrialBetw),
                        WedgeApertures[::-1], WedgeApertures,
                        np.zeros(NrNullTrialEnd),
                        ))

# prepare to indicate weather flicker, expanding or contracting motion
# counterbalance occurance of
lst = list(itertools.permutations(np.arange(NrOfCond)+1, NrOfCond))
for ind, item in enumerate(lst):
    lst[ind] = list(np.hstack(item))
aryPerm = np.array(lst)

# loop through combinations
for ind, indCond in enumerate(aryPerm):
    aperture = np.hstack((np.zeros(NrNullTrialStart),
                          np.ones(2*len(BarApertures))*indCond[0],
                          np.zeros(NrNullTrialBetw),
                          np.ones(2*len(BarApertures))*indCond[1],
                          np.zeros(NrNullTrialBetw),
                          np.ones(2*len(WedgeApertures))*indCond[2],
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
                                     len(Conditions1)-NrNullTrialEnd),
                                     NrOfTargets,
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
    if 2*ind+1 > 9:
        filename1 = os.path.join(str_path_parent_up, 'Conditions',
                                 'Conditions_MotLoc_run' + str(2*ind+1))
    else:
        filename1 = os.path.join(str_path_parent_up, 'Conditions',
                                 'Conditions_MotLoc_run0' + str(2*ind+1))
    np.savez(filename1, Conditions=Conditions1,
             TargetTRs=Targets, TargetOnsetinSec=TargetOnsetinSec,
             TargetDuration=TargetDuration, ExpectedTR=ExpectedTR)
    if 2*ind+2 > 9:
        filename2 = os.path.join(str_path_parent_up, 'Conditions',
                                 'Conditions_MotLoc_run' + str(2*ind+2))
    else:
        filename2 = os.path.join(str_path_parent_up, 'Conditions',
                                 'Conditions_MotLoc_run0' + str(2*ind+2))
    np.savez(filename2, Conditions=Conditions2,
             TargetTRs=Targets, TargetOnsetinSec=TargetOnsetinSec,
             TargetDuration=TargetDuration, ExpectedTR=ExpectedTR)
