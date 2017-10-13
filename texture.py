#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:07:50 2017

@author: Marian
"""

import numpy as np

pixel = 0.1
size = 48
sigma = 0.05


texture = np.empty((size, size))

def makeLoG(size, sigma, pixel):
    x, y = np.meshgrid(np.linspace(-pixel, pixel, size),
                       np.linspace(-pixel, pixel, size))
    part1 = (1/sigma**2)
    part2 = (1-((x**2 + y**2)/sigma**2)) 
    part3 = np.exp(-(x**2 + y**2)/(2*sigma**2))
    return part1 * part2 * part3

texture = makeLoG(size, sigma, pixel)

def d3_scale(dat, out_range=(-1, 1)):
    origShape= dat.shape
    dat = dat.flatten()
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat)).reshape(origShape)

texture = d3_scale(texture, out_range=(-1, 1))
