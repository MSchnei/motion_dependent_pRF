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

# %% derive paramters

# derive the number of combinations
nrOfCombis = len(cfg.combis)
# derive the number of position shifts
nrOfPosShifts = len(cfg.posShifts)
# derive the number of eccentricities
nrOfEcc = len(cfg.ecc)

combiInd = range(nrOfCombis)
posShiftsInd = range(nrOfPosShifts)
eccInd = range(nrOfEcc)

iterables = [combiInd, posShiftsInd, eccInd]
conditions = list(itertools.product(*iterables))
# unpack the zipping
for ind, item in enumerate(conditions):
    conditions[ind] = list(np.hstack(item))
conditions = np.array(conditions)


# %% save the results
for i in range(8):
    np.random.shuffle(conditions)
    strPathParentUp = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..'))
    if i > 9:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_Psychophysics_run' + str(i+1))
    else:
        filename1 = os.path.join(strPathParentUp, 'Conditions',
                                 'Conditions_Psychophysics_run0' + str(i+1))
    np.savez(filename1, conditions=conditions.astype('int8'))
