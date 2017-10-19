#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:46:57 2017

@author: marian
"""

import numpy as np
from psychopy import visual, core, event, logging
import MotionClouds as mc
import os

# width and height of your screen
w, h = 1920, 1200
#w, h = 2560, 1440

# width and height of the stimulus
w_stim, h_stim = 1024, 1024

loops = 2

# %% Motion clouds
# V_X and V_Y determine the speed in x and y direction
# (V_X, V_Y) = (0,1) is downward and  (V_X, V_Y) = (1, 0) is rightward
# A speed of V_X=1 corresponds to an average displacement of 1/N_X per frame
# B_V determines the bandwidth fo the speed

# theta determines the mean orientation (in radians, von-Mises distribution)
# B_theta is the bandwidth of the orientation (in radians)
# (standard deviation of the Gaussian envelope; HWHH = 2*B_theta_**2*np.log(2))

# sf_0 determines the preferred spatial frequency
# B_sf determines the bandwidth of spatial frequency

fx, fy, ft = mc.get_grids(mc.N_X, mc.N_Y, mc.N_frame)
seed = 1234
size = 5
N_X, N_Y, N_frame = 2**size, 2**size, mc.N_frame

fx, fy, ft = mc.get_grids(N_X, N_Y, N_frame)

N_orient = 9

# %%
x, y = np.meshgrid(np.arange(-N_orient/2., N_orient/2.)+0.5,
                   np.arange(-N_orient/2., N_orient/2.)+0.5)

def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
    return t, r

def pol2cart(r, t):
    x = r * np.cos(t)
    y = r * np.sin(t)
    return(x, y)

# if necessary scale the vector length
theta, radius = cart2pol(x, y)
# make vector unit length everywhere
radius = radius/radius
x, y = pol2cart(radius, theta)

x = x.reshape(-1, order='F')
y = y.reshape(-1, order='F')

# %%



# %%
hzn1 = mc.np.zeros((N_orient*N_X, N_orient*N_X, N_frame))
for i, x_i in enumerate(np.linspace(-1, 1., N_orient)):
    for j, x_j in enumerate(np.linspace(-1, 1., N_orient)):
        V_X = 2 * x_i / (1+x_i**2)
        V_Y = 2 * x_j / (1+x_j**2)

        # f_0 = ...
        # theta = np.arctan2(V_Y, V_X)
        env = mc.envelope_gabor(fx, fy, ft, V_X=V_X, V_Y=V_Y,
                                B_theta=np.inf, B_sf=np.inf)
        speed2 = mc.random_cloud(env, seed=seed)
        hzn1[i*N_X:(i+1)*N_X, j*N_Y:(j+1)*N_Y, :] = speed2

# %%

# since mc.rectif brings values in range 0 to 1, we need to multiply by 2 and
# subtract 1 in order to get range -1 to 1, which psychopy needsZ
z = 2*mc.rectif(hzn1, contrast=1.0) - 1.

# reverse z in third dimension
z = z[:, :, ::-1]

# scramble z in third dimension to generate the control
scramble = np.random.permutation(z.shape[2])
z = z[:, :, scramble]

# %% Psychopy
logging.console.setLevel(logging.DEBUG)

win = visual.Window([w, h], fullscr=True)
stim = visual.GratingStim(
    win,
    size=(w_stim, h_stim), units='pix',
    interpolate=True,
    mask='none',
    autoLog=False)

core.wait(1)

for i_frame in range(mc.N_frame * loops):
    # creating a new stimulus every time
    stim.tex = z[:, :, i_frame % mc.N_frame]
    stim.draw()
    win.flip()

win.close()
