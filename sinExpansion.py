# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:46:04 2017

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
from psychopy import visual, event, core,  monitors


dim = 1024
nFrames = 60

x, y = np.meshgrid(np.arange(-dim/2., dim/2.)+0.5,
                   np.arange(-dim/2., dim/2.)+0.5)


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
    return t, r

# if necessary scale the vector length
theta, radius = cart2pol(x, y)

# set monitor information:
distanceMon = 99  # [99 for Nova coil]
widthMon = 30  # [30 for Nova coil]
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in scanner

phase = np.linspace(0., 4.*np.pi, nFrames)

noiseTexture = np.zeros((dim, dim, nFrames))

for ind, t in enumerate(phase):

    ima = np.sin(0.05 * radius - t)
    noiseTexture[..., ind] = ima


# %%
"""FUNCTIONS"""
# update flicker in a square wave fashion
# with every frame
from itertools import cycle
from scipy import signal

raisedCos = signal.hann(nFrames/2.)[:nFrames/4.]

tempArray = np.zeros(nFrames)
tempArray[nFrames/4.:-nFrames/4.] = 1
tempArray[:nFrames/4.] = raisedCos
tempArray[-nFrames/4.:] = raisedCos[::-1]

tempCycle = cycle(tempArray)


def squFlicker():
    mContrast = tempCycle.next()
    return mContrast



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


movRTP = visual.GratingStim(
    myWin,
    tex=np.zeros((dim, dim)),
    mask='none',
    pos=(0.0, 0.0),
    size=(fieldSizeinPix, fieldSizeinPix),
    sf=None,
    ori=0.0,
    phase=(0.0, 0.0),
    color=(1.0, 1.0, 1.0),
    colorSpace='rgb',
    contrast=1.0,
    opacity=1.0,
    depth=0,
    rgbPedestal=(0.0, 0.0, 0.0),
    interpolate=False,
    name='movingRTP',
    autoLog=None,
    autoDraw=False,
    maskParams=None)

switch = True
ind = 0
while switch:
    print int(ind)
    movRTP.tex = noiseTexture[..., int(ind)]
    movRTP.contrast = squFlicker()
    movRTP.draw()
    myWin.flip()
    ind += 1
    ind = ind % nFrames
    # handle key presses each frame
    for key in event.getKeys():
        if key in ['escape', 'q']:
            switch = False
            myWin.close()
            core.quit()