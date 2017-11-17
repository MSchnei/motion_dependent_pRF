# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:58:06 2017

@author: marian
"""

from psychopy import visual, event, core, monitors

# %% MONITOR AND WINDOW

# set monitor information:
distanceMon = 29.5  # [99 for Nova coil]
widthMon = 35  # [30 for Nova coil]
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in scanner

moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

# set screen:
myWin = visual.Window(
    size=(PixW, PixH),
    screen=0,
    winType='pyglet',
    allowGUI=False,
    allowStencil=False,
    fullscr=True,
    monitor=moni,
    color=[0, 0, 0],
    colorSpace='rgb',
    units='deg',
    blendMode='avg',
    autoLog=False)

# make two wedges (in opposite contrast) and alternate them for flashing
wedge1 = visual.RadialStim(myWin, tex='sqrXsqr', color=1, size=24,
                           visibleWedge=[0, 360], radialCycles=12,
                           angularCycles=16, interpolate=False, autoLog=False)
wedge2 = visual.RadialStim(myWin, tex='sqrXsqr', color=-1, size=24,
                           visibleWedge=[0, 360], radialCycles=12,
                           angularCycles=16, interpolate=False, autoLog=False)
t = 0
clock = core.Clock()

# seconds for one B-W cycle (ie 1/Hz)
flashPeriod = 0.5
while True:
    t = clock.getTime()
    if (t % flashPeriod) < (flashPeriod / 2.0):
        stim = wedge1
    else:
        stim = wedge2

    stim.draw()
    myWin.flip()

    # Break out of the while loop if these keys are registered
    if event.getKeys(keyList=['escape', 'q']):
        break

myWin.close()
