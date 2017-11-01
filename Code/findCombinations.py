# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:33:42 2017

@author: marian
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt

# set number of repetitions (number of times the stimulus is shows)
n = 5
lst = list(itertools.product([0, 1], repeat=n))

for ind, item in enumerate(lst):
    lst[ind] = list(np.hstack(item))
lst = np.array(lst)

# set number of successes (number of times the area is on)
numSuc = 3
success = np.copy(lst[np.sum(lst, axis=1) == numSuc, :])

aryCorr = np.empty((len(success), len(success)))

for ind in range(success.shape[0]):
    # get row
    row = success[ind, :]
    # get other rows
    otherrowind = np.where(np.arange(success.shape[0]) != ind)[0]
    otherrows = success[otherrowind, :]
    for ind2 in otherrowind:
        # get one of the other rows
        otherrow = success[ind2, :]
        # cxalculate correlations
        # put away in matrix
        aryCorr[ind, ind2] = np.sum(row == otherrow)

plt.imshow(aryCorr)


#aryPairs = np.empty((len(success), 4))
#
#for ind in range(aryCorr.shape[0]):
#    row = aryCorr[ind, :]
#    newarray = np.array(ind)
#    aryPairs[ind, :] = np.sort(np.hstack((newarray, np.where(row==1)[0])))
#
#
#aryCorr2 = np.empty((len(aryPairs), len(aryPairs)))
#for ind in range(aryPairs.shape[0]):
#    # get row
#    row = aryPairs[ind, :]
#    # get other rows
#    otherrowind = np.where(np.arange(aryPairs.shape[0]) != ind)[0]
#    otherrows = aryPairs[otherrowind, :]
#    for ind2 in otherrowind:
#        # get one of the other rows
#        otherrow = aryPairs[ind2, :]
#        # cxalculate correlations
#        # put away in matrix
#        aryCorr2[ind, ind2] = len(np.intersect1d(row, otherrow))