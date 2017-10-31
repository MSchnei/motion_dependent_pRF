#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:27:18 2017

@author: Marian
"""

import numpy as np

size = 5
N_X, N_Y, N_frame = 2**size, 2**size, 128
N_fields = 9
V_X = []
V_Y = []
for i, x_i in enumerate(np.linspace(-1, 1., N_fields)):
    for j, x_j in enumerate(np.linspace(-1, 1., N_fields)):
        V_X.append(2 * x_i / (1+x_i**2))
        V_Y.append(2 * x_j / (1+x_j**2))

x, y = np.meshgrid(np.arange(-N_fields/2., N_fields/2.)+0.5,
                   np.arange(-N_fields/2., N_fields/2.)+0.5)


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
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





