# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:48:56 2017

@author: marian
"""

import numpy as np
import re
import itertools
import os
import matplotlib.pyplot as plt


# set the number of line where the keypress logging starts
docStart = 201.

path = "/media/sf_D_DRIVE/MotDepPrf/PsychoPhysics/S01/Logging"

lstNames = [
    "P01_psychophycis_motDepPrf_Run01_2017_Nov_15_1843.log",
    "P01_psychophycis_motDepPrf_Run02_2017_Nov_15_1852.log",
    "P01_psychophycis_motDepPrf_Run03_2017_Nov_15_1902.log",
    "P01_psychophycis_motDepPrf_Run04_2017_Nov_15_1912.log",
    ]

lstPresses = []
lstCond = []

for idx in range(len(lstNames)):
    # deduce name of file correctd for white spaces
    filename = os.path.join(path, lstNames[idx])
    name = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(os.path.basename(filename))[1]
    file_name = name + "_cor" + ext
    filenameOut = os.path.join(path, file_name)

    # add and remove white spaces where necessary
    replacements = {'[': '[ ', ']': ' ]'}
    with open(filename) as infile, open(filenameOut, 'w') as outfile:
        for line in infile:
            for src, target in replacements.iteritems():
                line = line.replace(src, target)
                line = re.sub(' +', ' ', line)
            outfile.write(line)
    # read time and conditions
    with open(filenameOut) as f_in:
        conditions = np.genfromtxt(itertools.islice(f_in, 0, None, 2),
                                   comments='#',
                                   delimiter=' ',
                                   usecols=(0, 3, 4, 5),
                                   skip_header=int(docStart/2.),
                                   skip_footer=1)
    lstCond.append(conditions)

    # read time and key presses
    with open(filenameOut) as f_in:
        presses = np.genfromtxt(itertools.islice(f_in, 1, None, 2),
                                comments='#',
                                delimiter=' ',
                                usecols=(0, 3),
                                skip_header=int(docStart/2.),
                                skip_footer=0)
    lstPresses.append(presses)

# combine comnditions and reponses
resp = np.concatenate((np.concatenate(lstCond, axis=0)[:, 1:],
                       np.concatenate(lstPresses, axis=0)[:, 1:]), axis=1
                      ).astype(np.int)

# split data according to eccentricity
respEcc1 = resp[np.argsort(resp[:, 2])][:len(resp)/2.][:, [0, 1, 3]]
respEcc2 = resp[np.argsort(resp[:, 2])][len(resp)/2.:][:, [0, 1, 3]]

# replace response 2 with 0
respEcc1[respEcc1[:, 2] == 2, 2] = 0
respEcc2[respEcc2[:, 2] == 2, 2] = 0

# sort data by position shift and combi
respEcc1 = respEcc1[np.lexsort((respEcc1[:, 0], respEcc1[:, 1]))]
respEcc2 = respEcc2[np.lexsort((respEcc2[:, 0], respEcc2[:, 1]))]

# reshape and get only key responses
respEcc1 = respEcc1.reshape((11, 8, 4, 3))[..., 2]
respEcc2 = respEcc2.reshape((11, 8, 4, 3))[..., 2]

# simplify by grouping together opposite, same, up vs. stationary,
# down vs. stationary 
# For this to work, we need to ...

# 1) for every second column flip order of elemnts
respEcc1[:, 1::2, :] = respEcc1[::-1, 1::2, :]
respEcc2[:, 1::2, :] = respEcc2[::-1, 1::2, :]

# 2) for every second column, flip responses (0 -> 1; 1 -> 0)
lgcZeros = [respEcc1[:, 1::2, :] == 0][0]
lgcOnes = [respEcc1[:, 1::2, :] == 1][0]
respEcc1[:, 1::2, :][lgcZeros] = 1
respEcc1[:, 1::2, :][lgcOnes] = 0

lgcZeros = [respEcc2[:, 1::2, :] == 0][0]
lgcOnes = [respEcc2[:, 1::2, :] == 1][0]
respEcc2[:, 1::2, :][respEcc2] = 1
respEcc2[:, 1::2, :][respEcc2] = 0

# finally simplify by grouping neighboring columns in 2nd axis together
# then we have 11 psotions, only 4 combis and 8 repetitions
respEcc1 = respEcc1.reshape((11, 4, 8))
respEcc2 = respEcc2.reshape((11, 4, 8))

# get mean
meanrespEcc1 = np.mean(respEcc1, axis=2)
meanrespEcc2 = np.mean(respEcc2, axis=2)

meanResp = (meanrespEcc1 + meanrespEcc2)/2.

# plotting
x = np.array([-0.8, -0.4, -0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.4, 0.8])

plt.plot(x, meanResp[:, 1], color='g', marker='o', markersize=12,
         label='same')
plt.plot(x, meanResp[:, 2], color='b', marker='o', markersize=12,
         label='up-stat')
plt.plot(x, meanResp[:, 3], color='y', marker='o', markersize=12,
         label='down-stat')

plt.legend()
plt.show()
