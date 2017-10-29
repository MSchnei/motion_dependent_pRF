#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 16:55:45 2017

@author: marian
"""
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
from psychopy import visual, event, core,  monitors


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)


def doRotation(x, y, RotRad=0):
    """Generate a meshgrid and rotate it by RotRad radians."""

    # Clockwise, 2D rotation matrix
    RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                          [-np.sin(RotRad), np.cos(RotRad)]])

    rot = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))
    rot = np.transpose(rot, (1, 2, 0))
    return rot[..., 0], rot[..., 1]

def PrettyPattern(lamb, sptfrq, phase, numSquares, dim):
    """
    Draws a pretty pattern stimulus.
    
    Parameters:
        lambda
            Wavelength of the sinusoid
        sptfrq
            Spatial frequency of checkers
        phase
            Phase of the sinusoid
        numSquares
            Number of Squares in the image
    
    The function returns the new image.
    """

    # 1 square is equivalent to 360 degree input
    width = numSquares * 360
    # what matters is the diagonal, so shrink width with sqrt(2)/2
    width = width * np.sqrt(2)/2

    # get meshgrid of X and Y coordinates
    X, Y = np.meshgrid(np.linspace(-width/2, width/2, dim),
                       np.linspace(-width/2, width/2, dim))
    # rotate by 45 degress
    X, Y = doRotation(X, Y, RotRad=np.pi/4.)
    # set X[0, 0] to 270, since np.sin(270) = -1
    if np.greater_equal(X[0, 0], 0):
        X = X + X[0, 0] + 270
    else:
        X = X - X[0, 0] + 270
    # set Y[0, 0] to 0, since np.cos(0) = 1
    if np.greater_equal(Y[0, 0], 0):
        Y = Y + Y[0, 0]
    else:
        Y = Y - Y[0, 0]
    # Luminance modulation at each pixel
    nom = np.sin(((sptfrq*np.pi*X)/180.)) + np.cos(((sptfrq*np.pi*Y)/180.))
    img = (np.cos(2*np.pi*nom / lamb + phase))

    return img

# set monitor information:
distanceMon = 99  # [99 for Nova coil]
widthMon = 30  # [30 for Nova coil]
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in scanner

# replace the 4s with sptfrq??
phase = np.linspace(0., 4.*np.pi, 72.)
lamb = np.sin(phase)/4. + 0.5

dim = 1024
numSquares = 8

noiseTexture = np.zeros((dim, dim, 72))
X, Y = np.meshgrid(np.linspace(-PixH/2., PixH/2., dim, endpoint=False)+1,
                   np.linspace(-PixH/2., PixH/2., dim, endpoint=False)+1)

for ind, (t, d) in enumerate(zip(phase, lamb)):

    ima = PrettyPattern(d, 1, t, numSquares, dim)
    # 1 = white, #-1 = black
    ima[np.greater(ima, 0)] = 1
    ima[np.less_equal(ima, 0)] = -1
    noiseTexture[..., ind] = ima

## save array as images, if wanted
#from PIL import Image
#for ind in np.arange(noiseTexture.shape[2]):
#    im = Image.fromarray(noiseTexture[..., ind].astype(np.uint8))
#    im.save("/home/marian/Documents/Testing/carrierPattern/Ima" + "_" +
#            str(ind) + ".png")


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
    movRTP.draw()
    myWin.flip()
    ind += 0.5
    ind = ind % 72
    # handle key presses each frame
    for key in event.getKeys():
        if key in ['escape', 'q']:
            switch = False
            myWin.close()
            core.quit()
