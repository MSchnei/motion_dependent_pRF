# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:45:25 2017

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
from psychopy import visual, event, core,  monitors

dim = 512
fieldSizeinDeg = 10
nFrames = 120

x, y = np.meshgrid(np.linspace(-fieldSizeinDeg/2., fieldSizeinDeg/2., dim),
                   np.linspace(-fieldSizeinDeg/2., fieldSizeinDeg/2., dim))


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
    return t, r


# if necessary scale the vector length
theta, radius = cart2pol(x, y)
phase = np.linspace(0., 4.*np.pi, nFrames)
spatFreq = np.hstack((np.ones(nFrames/2.), np.ones(nFrames/2.) * 2))
# np.sin(phase) * 0.5 + 1


noiseTexture = np.zeros((dim, dim, nFrames))

#for ind, (t, fr) in enumerate(zip(phase, spatFreq)):
#
#    nom = np.sin((fieldSizeinDeg/2.) * fr * theta - t)
#    ima = (np.cos(2*np.pi*nom / fr + t))
#
#    noiseTexture[..., ind] = ima

for ind, (t, fr) in enumerate(zip(phase, spatFreq)):

    nom = np.sin((fieldSizeinDeg/2.) * fr * radius + t)
    ima = (np.cos(2*np.pi*nom / fr))

    noiseTexture[..., ind] = ima

# %%
"""MONITOR AND WINDOW"""
# set monitor as object
# set monitor information:
distanceMon = 99  # [99 for Nova coil]
widthMon = 30  # [30 for Nova coil]
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in scanner
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
    movRTP.draw()
    myWin.flip()
    ind += 0.5
    ind = ind % nFrames
    # handle key presses each frame
    for key in event.getKeys():
        if key in ['escape', 'q']:
            switch = False
            myWin.close()
            core.quit()
