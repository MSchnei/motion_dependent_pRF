#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:53:56 2017

@author: Marian
"""

import numpy as np


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    t = np.rad2deg(t)
    return t, r

def pol2cart(r, t):
    t = np.deg2rad(t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return(x, y)

def createBinCircleMask(size, rMin=0, rMax=500, thetaMin=-180, thetaMax=180):
    # create meshgrid
    x, y = np.meshgrid(np.arange(-size/2., size/2.)+0.5,
                   np.arange(-size/2., size/2.)+0.5)
    # convert to polar coordinates
    theta, radius = cart2pol(x, y)
    # define ringMask
    ringMask = np.logical_and(np.greater(radius, rMin),
                              np.less_equal(radius, rMax))
    
    # define wedgeMask
    wedgeMask = np.logical_and(np.greater(theta, thetaMin),
                               np.less_equal(theta, thetaMax))
    
    
    return np.logical_and(ringMask, wedgeMask)


    
test = createBinCircleMask(1200, rMin=0, rMax=600, thetaMin=-120, thetaMax=180)
    
    