# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:15:09 2017

@author: marian
"""

"""FUNCTIONS"""
# update flicker in a square wave fashion
# with every frame
from itertools import cycle
from scipy import signal
import numpy as np


nFrames = 120
raisedCos = signal.hann(nFrames/2.)[:nFrames/4.]

tempArray = np.zeros(nFrames)
tempArray[nFrames/4.:-nFrames/4.] = 1
tempArray[:nFrames/4.] = raisedCos
tempArray[-nFrames/4.:] = raisedCos[::-1]

tempCycle = cycle(tempArray)


def squFlicker():
    mContrast = tempCycle.next()
    return mContrast

#test = np.tile(tempArray, 10)
#plt.plot(test)
#axes = plt.gca()
#axes.set_ylim([0,1.1])
#plt.savefig("/home/marian/gdrive/Research/MotionLocaliser/ProjectProposal/RaisedCos.svg")
