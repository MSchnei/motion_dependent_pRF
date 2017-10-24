#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:55:45 2017

@author: marian
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
from psychopy import visual, event, core,  monitors


# set monitor information:
distanceMon = 99  # [99 for Nova coil]
widthMon = 30  # [30 for Nova coil]
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in scanner


# %%
conditions = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
durations = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# %%
"""MONITOR AND WINDOW"""
# set monitor as object
moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

fieldSizeinPix = 1024

# set screen:
# for psychoph lab: make 'fullscr = True', set size =(1920, 1080)
myWin = visual.Window(
    size=(PixW, PixH),
    screen=0,
    winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
    allowGUI=False,
    allowStencil=True,
    fullscr=False,  # for psychoph lab: fullscr = True
    monitor=moni,
    color=[0, 0, 0],
    colorSpace='rgb',
    units='pix',
    blendMode='avg')

circle00 = visual.Circle(myWin, radius=0.0, edges=64, units='deg', lineWidth=0,
                         fillColor=(-1.0, -1.0, -1.0), fillColorSpace='rgb')

# %%
"""TIME, TIMING AND CLOCKS"""
# parameters
totalTime = np.sum(durations)

# define clock
clock = core.Clock()

# %%
"""FUNCTIONS"""
# update flicker in a square wave fashion
# with every frame
from itertools import cycle
numFrame = 10
squareArray = np.sin(np.linspace(0, np.pi/2., numFrame))
squareCycle = cycle(squareArray)

def squFlicker():
    mContrast = squareCycle.next()
    return mContrast


# %%
"""RENDER_LOOP"""
# Create Counters
i = 0  # counter for blocks

sequenceExp = np.linspace(0, 0.5, 10, endpoint=False)
#test = np.sin(np.linspace(0, np.pi, 10))
#test = (test / np.sum(test)) * 0.5
#sequenceExp = np.cumsum(test)

sequenceCon = np.copy(sequenceExp[::-1])

sequenceControl = np.copy(sequenceExp)
sequenceControl[:] = 0

fill1 = [-1.0, -1.0, -1.0]
fill2 = [1.0, 1.0, 1.0]

while clock.getTime() < totalTime:

    # expanding
    if conditions[i] == 0:
        sequence = np.copy(sequenceExp)

    # contracting
    elif conditions[i] == 1:
        sequence = np.copy(sequenceCon)

    # control1
    elif conditions[i] == 2:
        sequence = np.copy(sequenceControl)

    while clock.getTime() < np.sum(durations[0:i+1]):

        for t, ind in enumerate(sequence):
            print ind

            circle00.radius = 5
            circle00.fillColor = fill1
            circle00.draw()
            circle00.radius = 4.5 + ind
            circle00.fillColor = fill2
            circle00.draw()
            circle00.radius = 4.0 + ind
            circle00.fillColor = fill1
            circle00.draw()
            circle00.radius = 3.5 + ind
            circle00.fillColor = fill2
            circle00.draw()
            circle00.radius = 3.0 + ind
            circle00.fillColor = fill1
            circle00.draw()
            circle00.radius = 2.5 + ind
            circle00.fillColor = fill2
            circle00.draw()
            circle00.radius = 2.0 + ind
            circle00.fillColor = fill1
            circle00.draw()
            circle00.radius = 1.5 + ind
            circle00.fillColor = fill2
            circle00.draw()
            circle00.radius = 1.0 + ind
            circle00.fillColor = fill1
            circle00.draw()
            circle00.radius = 0.5 + ind
            circle00.fillColor = fill2
            circle00.draw()
            circle00.radius = 0.0 + ind
            circle00.fillColor = fill1
            circle00.draw()

            myWin.flip()

            if t == len(sequence)-1:
                print "Change contrast"
                fill1 = np.array(fill1)*-1
                fill1 = list(fill1)
                fill2 = np.array(fill2)*-1
                fill2 = list(fill2)

            # handle key presses each frame
            for key in event.getKeys():
                if key in ['escape', 'q']:
                    myWin.close()
                    core.quit()
    i = i+1

myWin.close()
core.quit()
