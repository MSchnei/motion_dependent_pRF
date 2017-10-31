# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:59:09 2017

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
"""MONITOR AND WINDOW"""
# set monitor as object
moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

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


movRTP = visual.RadialStim(
    myWin,
    tex='sqrXsqr',
    mask='Gauss',
    units='deg',
    pos=(0.0, 0.0),
    size=(5.0, 5.0),
    radialCycles=5,
    angularCycles=0,
    radialPhase=0,
    angularPhase=0,
    ori=0.0,
    texRes=64,
    angularRes=100,
    visibleWedge=(0, 360),
    rgb=None, color=(1.0, 1.0, 1.0),
    colorSpace='rgb',
    dkl=None,
    lms=None,
    contrast=1.0,
    opacity=1.0,
    depth=0,
    rgbPedestal=(0.0, 0.0, 0.0),
    interpolate=False,
    name=None,
    autoLog=None,
    maskParams=None)

# fixation dot
dotFix = visual.Circle(
    myWin,
    units='deg',
    name='dotFix',
    radius=2.0,
    fillColor=[0.0, 0.0, 0.0],
    lineColor=[0.0, 0.0, 0.0],)

switch = True
ind = 0
while switch:

    ind += 0.5
    ind = ind % 10
    if np.isclose(ind, 0):
            movRTP.contrast *= -1
    print int(ind)
    movRTP.draw()
    dotFix.draw()
    myWin.flip()

    # handle key presses each frame
    for key in event.getKeys():
        if key in ['escape', 'q']:
            switch = False
            myWin.close()
            core.quit()
