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
conditions = np.array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
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
circle01 = visual.Circle(myWin, radius=0.5, edges=64, units='deg', lineWidth=0,
                         fillColor=(1.0, 1.0, 1.0), fillColorSpace='rgb')
circle02 = visual.Circle(myWin, radius=1.0, edges=64, units='deg', lineWidth=0,
                         fillColor=(-1.0, -1.0, -1.0), fillColorSpace='rgb')
circle03 = visual.Circle(myWin, radius=1.5, edges=64, units='deg', lineWidth=0,
                         fillColor=(1.0, 1.0, 1.0), fillColorSpace='rgb')
circle04 = visual.Circle(myWin, radius=2.0, edges=64, units='deg', lineWidth=0,
                         fillColor=(-1.0, -1.0, -1.0), fillColorSpace='rgb')
circle05 = visual.Circle(myWin, radius=2.5, edges=64, units='deg', lineWidth=0,
                         fillColor=(1.0, 1.0, 1.0), fillColorSpace='rgb')
circle06 = visual.Circle(myWin, radius=3.0, edges=64, units='deg', lineWidth=0,
                         fillColor=(-1.0, -1.0, -1.0), fillColorSpace='rgb')
circle07 = visual.Circle(myWin, radius=3.5, edges=64, units='deg', lineWidth=0,
                         fillColor=(1.0, 1.0, 1.0), fillColorSpace='rgb')
circle08 = visual.Circle(myWin, radius=4.0, edges=64, units='deg', lineWidth=0,
                         fillColor=(-1.0, -1.0, -1.0), fillColorSpace='rgb')
circle09 = visual.Circle(myWin, radius=4.5, edges=64, units='deg', lineWidth=0,
                         fillColor=(1.0, 1.0, 1.0), fillColorSpace='rgb')
circle10 = visual.Circle(myWin, radius=5.0, edges=64, units='deg', lineWidth=0,
                         fillColor=(-1.0, -1.0, -1.0), fillColorSpace='rgb')

# %%
"""TIME, TIMING AND CLOCKS"""
# parameters
totalTime = np.sum(durations)

# define clock
clock = core.Clock()

# %%
"""RENDER_LOOP"""
# Create Counters
i = 0  # counter for blocks

sequenceExp = np.linspace(0, 0.5, 10, endpoint=False)
sequenceCon = np.copy(sequenceExp[::-1])
sequenceControl = np.copy(sequenceExp)
np.random.shuffle(sequenceControl)


switch = True
while clock.getTime() < totalTime:

    # expanding
    if conditions[i] == 0:
        sequence = np.copy(sequenceExp)

    # contracting
    elif conditions[i] == 1:
        sequence = np.copy(sequenceCon)

    # control
    elif conditions[i] == 2:
        sequence = np.copy(sequenceControl)

    while clock.getTime() < np.sum(durations[0:i+1]):

        for ind in sequence:
            print ind

            circle00.radius = 0.0 + ind
            circle01.radius = 0.5 + ind
            circle02.radius = 1.0 + ind
            circle03.radius = 1.5 + ind
            circle04.radius = 2.0 + ind
            circle05.radius = 2.5 + ind
            circle06.radius = 3.0 + ind
            circle07.radius = 3.5 + ind
            circle08.radius = 4.0 + ind
            circle09.radius = 4.5 + ind

            circle10.draw()
            circle09.draw()
            circle08.draw()
            circle07.draw()
            circle06.draw()
            circle05.draw()
            circle04.draw()
            circle03.draw()
            circle02.draw()
            circle01.draw()
            circle00.draw()

            myWin.flip()

            if np.isclose(ind, sequence[-1]):
                print "Change contrast"
                circle10.fillColor *= -1
                circle09.fillColor *= -1
                circle08.fillColor *= -1
                circle07.fillColor *= -1
                circle06.fillColor *= -1
                circle05.fillColor *= -1
                circle04.fillColor *= -1
                circle03.fillColor *= -1
                circle02.fillColor *= -1
                circle01.fillColor *= -1
                circle00.fillColor *= -1

            # handle key presses each frame
            for key in event.getKeys():
                if key in ['escape', 'q']:
                    myWin.close()
                    core.quit()
    i = i+1

myWin.close()
core.quit()
