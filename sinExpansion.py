# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:46:04 2017

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
from scipy import signal
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
from utils import (cart2pol, time2frame, raisedCos)

# %%
""" SAVING and LOGGING """
# Store info about experiment and experimental run
expName = 'prfStim_Motion'  # set experiment name here
expInfo = {
    u'maskType': ['mskCircleBar', 'mskSquare', 'mskBar', 'mskCircle'],
    u'participant': u'pilot',
    u'run': u'01',
    }
# Create GUI at the beginning of exp to get more expInfo
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK is False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# get current path and save to variable _thisDir
_thisDir = os.path.dirname(os.path.abspath(__file__))
# get parent path and move up one directory
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
# move to parent_up path
os.chdir(str_path_parent_up)

# Name and create specific subject folder
subjFolderName = str_path_parent_up + os.path.sep + \
    '%s_SubjData' % (expInfo['participant'])
if not os.path.isdir(subjFolderName):
    os.makedirs(subjFolderName)
# Name and create data folder for the experiment
dataFolderName = subjFolderName + os.path.sep + '%s' % (expInfo['expName'])
if not os.path.isdir(dataFolderName):
    os.makedirs(dataFolderName)
# Name and create specific folder for logging results
logFolderName = dataFolderName + os.path.sep + 'Logging'
if not os.path.isdir(logFolderName):
    os.makedirs(logFolderName)
logFileName = logFolderName + os.path.sep + '%s_%s_Run%s_%s' % (
    expInfo['participant'], expInfo['expName'],
    expInfo['run'], expInfo['date'])
# Name and create specific folder for pickle output
outFolderName = dataFolderName + os.path.sep + 'Output'
if not os.path.isdir(outFolderName):
    os.makedirs(outFolderName)
outFileName = outFolderName + os.path.sep + '%s_%s_Run%s_%s' % (
    expInfo['participant'], expInfo['expName'],
    expInfo['run'], expInfo['date'])

# save a log file and set level for msg to be received
logFile = logging.LogFile(logFileName+'.log', level=logging.INFO)
logging.console.setLevel(logging.WARNING)  # set console to receive warnings

# %%
"""MONITOR AND WINDOW"""
# set monitor information:
distanceMon = 99  # [99 for Nova coil]
widthMon = 30  # [30 for Nova coil]
PixW = 1920.0  # [1920.0] in scanner
PixH = 1200.0  # [1200.0] in scanner

moni = monitors.Monitor('testMonitor', width=widthMon, distance=distanceMon)
moni.setSizePix([PixW, PixH])  # [1920.0, 1080.0] in psychoph lab

# log monitor info
logFile.write('MonitorDistance=' + unicode(distanceMon) + 'cm' + '\n')
logFile.write('MonitorWidth=' + unicode(widthMon) + 'cm' + '\n')
logFile.write('PixelWidth=' + unicode(PixW) + '\n')
logFile.write('PixelHeight=' + unicode(PixH) + '\n')

# set screen:
myWin = visual.Window(
    size=(PixW, PixH),
    screen=0,
    winType='pyglet',
    allowGUI=False,
    allowStencil=True,
    fullscr=True,
    monitor=moni,
    color=[0, 0, 0],
    colorSpace='rgb',
    units='pix',
    blendMode='avg')

# The size of the field.
fieldSizeinDeg = 10
fieldSizeinPix = np.round(misc.deg2pix(fieldSizeinDeg, moni))

logFile.write('fieldSizeinDeg=' + unicode(fieldSizeinDeg) + '\n')
logFile.write('fieldSizeinPix=' + unicode(fieldSizeinPix) + '\n')

# %%
"""DURATIONS"""
# get timings for apertures and motion directions
Conditions = np.array(
    [[0, 11, 22, 33, 44, 55, 63, 4, 9, 40, 12, 24, 31, 45, 50, 64, 6, 12, 41],
     [0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]]).T
Conditions = Conditions.astype(int)

# get timings for the targets
targets = np.array([2, 4, 6, 8, 10])
# set expected TR
ExpectedTR = 2

# create array to log key pressed events
TriggerPressedArray = np.array([])
TargetPressedArray = np.array([])

# %%
"""TEXTURE AND MASKS"""

# define the texture
dim = 1024
nFrames = 60
cycPerSec = 2.5

# get cartesian coordinates which are needed to define the textures
x, y = np.meshgrid(np.linspace(-fieldSizeinDeg/2., fieldSizeinDeg/2., dim),
                   np.linspace(-fieldSizeinDeg/2., fieldSizeinDeg/2., dim))

# get polar coordinates which are needed to define the textures
theta, radius = cart2pol(x, y)
# define the phase for inward/outward conditin
phase = np.linspace(0., 2.*np.pi, nFrames/cycPerSec)
# define the spatial frequency of the radial sine wave grating
spatFreq = 2
angularCycles = 32

# get the array that divides field in angular cycles
polCycles = np.sin(angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1

# get radial sine wave gratings for main conditions
stimTexture = np.zeros((dim, dim, nFrames/cycPerSec))
for ind, ph in enumerate(phase):
    ima = np.sin((fieldSizeinDeg/2.) * spatFreq * radius - ph)
    stimTexture[..., ind] = ima

# get radial sine wave gratings for control condition
ctrlTexture = np.zeros((dim, dim, 2))
ima = np.sin((fieldSizeinDeg/2.) * spatFreq * radius)
ima = ima * polCycles
ctrlTexture[..., 0] = np.copy(ima)
ctrlTexture[..., 1] = np.copy(ima) * -1

# binarize
stimTexture[np.greater_equal(stimTexture, 0)] = 1
stimTexture[np.less(stimTexture, 0)] = -1
stimTexture = stimTexture.astype('int8')

ctrlTexture[np.greater_equal(ctrlTexture, 0)] = 1
ctrlTexture[np.less(ctrlTexture, 0)] = -1
ctrlTexture = ctrlTexture.astype('int8')

# retrieve the different masks
opaPgDnMasks = np.load("/home/marian/Documents/Testing/CircleBarApertures/" +
                       "opa/opaPgDnMasks.npy").astype('int32')
opaPgUpMasks = np.load("/home/marian/Documents/Testing/CircleBarApertures/" +
                       "opa/opaPgUpMasks.npy").astype('float32')
midGreyTexture = np.zeros((dim, dim))

# %%
"""STIMULI"""
# main stimulus
radSqrWave = visual.GratingStim(
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
    name='radSqrWave',
    autoLog=None,
    autoDraw=False,
    maskParams=None)

radSqrWaveBckgr = visual.GratingStim(
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
    name='radSqrWave',
    autoLog=None,
    autoDraw=False,
    maskParams=None)

# fixation dot
dotFix = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.125,
    fillColor=[1.0, 0.0, 0.0],
    lineColor=[1.0, 0.0, 0.0],)

# surround of the fixation dot
dotFixSurround = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFixSurround',
    units='deg',
    radius=0.19,
    fillColor=[1.0, 1.0, 1.0],
    lineColor=[1.0, 1.0, 1.0],)

# fixation grid circle
Circle = visual.Polygon(
    win=myWin,
    name='Circle',
    edges=90,
    ori=0,
    units='deg',
    pos=[0, 0],
    lineWidth=2,
    lineColor=[-0.2, -0.2, -0.2],
    lineColorSpace='rgb',
    fillColor=None,
    fillColorSpace='rgb',
    opacity=1,
    interpolate=True,
    autoLog=False,)

# fixation grid line
Line = visual.Line(
    win=myWin,
    name='Line',
    autoLog=False,
    start=(-PixH, 0),
    end = (PixH, 0),
    pos=[0, 0],
    lineWidth=2,
    lineColor=[-0.2, -0.2, -0.2],
    lineColorSpace='rgb',
    fillColor=None,
    fillColorSpace='rgb',
    opacity=1,
    interpolate=True,)

# initialisation method
message = visual.TextStim(
    myWin,
    text='Condition',
    height=30,
    pos=(400, 400)
    )

triggerText = visual.TextStim(
    win=myWin,
    color='white',
    height=30,
    text='Experiment will start soon.',)

targetText = visual.TextStim(
    win=myWin,
    color='white',
    height=30,
    autoLog=False,
    )

# %%
"""TIME AND TIMING PARAMETERS"""

# get screen refresh rate
refr_rate = myWin.getActualFrameRate()  # get screen refresh rate
if refr_rate is not None:
    frameDur = 1.0/round(refr_rate)
else:
    frameDur = 1.0/nFrames  # couldn't get a reliable measure so guess
print "refr_rate:"
print refr_rate
logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

# set durations
nrOfVols = len(Conditions)
durations = np.ones(nrOfVols)*2
totalTime = ExpectedTR*nrOfVols


# define on/off cycle in ms
lenCyc = 200.
# derive how much of second that is
div = 1000/lenCyc

# define array to cycle opacity
cycOpa = np.hstack((signal.hamming(nFrames/div)[:nFrames/(div*2)],
                    np.ones(nFrames/div),
                    signal.hamming(2*nFrames/(div*3))[nFrames/(div*3):],
                    np.zeros(nFrames/(div*3)),
                    signal.hamming(2*nFrames/(div*3))[:nFrames/(div*3)],
                    np.ones(nFrames/div),
                    signal.hamming(2*nFrames/(div*3))[nFrames/(div*3):],
                    np.zeros(nFrames/(div*3)),
                    signal.hamming(2*nFrames/(div*3))[:nFrames/(div*3)],
                    np.ones(nFrames/div),
                    signal.hamming(nFrames/div)[nFrames/(div*2):],
                    np.zeros(nFrames-nFrames/div))
                   ).astype('float32')
#cycOpa = np.ones(2*nFrames).astype('float32')

# create clock
clock = core.Clock()
logging.setDefaultClock(clock)

# %%
"""FUNCTIONS"""
def fixationGrid():
    Circle.setSize((2, 2))
    Circle.draw()
    Circle.setSize((4, 4))
    Circle.draw()
    Circle.setSize((6, 6))
    Circle.draw()
    Circle.setSize((8, 8))
    Circle.draw()
    Circle.setSize((10, 10))
    Circle.draw()
    Line.setOri(0)
    Line.draw()
    Line.setOri(45)
    Line.draw()
    Line.setOri(90)
    Line.draw()
    Line.setOri(135)
    Line.draw()

# %%
"""RENDER_LOOP"""
# Create Counters
i = 0
# give the system time to settle
core.wait(1)

# wait for scanner trigger
triggerText.draw()
myWin.flip()
event.waitKeys(keyList=['5'], timeStamped=False)

# reset clocks
clock.reset()
logging.data('StartOfRun' + unicode(expInfo['run']))

while clock.getTime() < totalTime:

    # get key for masks
    keyMask = Conditions[i, 0]
    # get mask to define the opacity values (foreground)
    opaPgUpMask = np.squeeze(opaPgUpMasks[:, :, keyMask])
    # get mask to define the opacity values (background)
    opaPgDnMask = np.squeeze(opaPgDnMasks[:, :, keyMask])

    # set the background mask to opaPgDnMask
    radSqrWaveBckgr.mask = opaPgDnMask

    # static/flicker control
    if Conditions[i, 1] == 0:
        tempIt = np.tile(
            np.repeat(np.array([0, 1]), nFrames/(cycPerSec*2)),
            cycPerSec*2).astype('int32')
        visTexture = ctrlTexture

    # contracting motion
    elif Conditions[i, 1] == 1:
        tempIt = np.tile(
            np.arange(nFrames/cycPerSec), cycPerSec*2).astype('int32')[::-1]
        visTexture = stimTexture

    # expanding motion
    elif Conditions[i, 1] == 2:
        tempIt = np.tile(
            np.arange(nFrames/cycPerSec), cycPerSec*2).astype('int32')
        visTexture = stimTexture

    while clock.getTime() < np.sum(durations[0:i+1]):
        # get interval time
        t = clock.getTime() % ExpectedTR
        # get respective frame
        frame = time2frame(t, frameRate=nFrames)

        # draw fixation grid (circles and lines)
        fixationGrid()

        # set opacity of background aperture
        radSqrWaveBckgr.opacity = cycOpa[int(frame)]
        # draw the background aperture
        radSqrWaveBckgr.draw()

        # set the foreground aperture
        radSqrWave.tex = visTexture[..., tempIt[int(frame)]]
        # set opacity of foreground aperture
        radSqrWave.opacity = cycOpa[int(frame)]
        # set foreground mask to opaPgDnMask
        radSqrWave.mask = opaPgUpMask
        # draw the foreground aperture
        radSqrWave.draw()

        # decide whether to draw target
        # first time in target interval? reset target counter to 0!
        if (sum(clock.getTime() >= targets) + sum(clock.getTime() <
           targets + 0.3) == len(targets)+1):
            # display target!
            # change color fix dot surround to red
            dotFix.fillColor = [0.5, 0.0, 0.0]
            dotFix.lineColor = [0.5, 0.0, 0.0]
        else:
            # dont display target!
            # keep color fix dot surround yellow
            dotFix.fillColor = [1.0, 0.0, 0.0]
            dotFix.lineColor = [1.0, 0.0, 0.0]

        # draw fixation point surround
        dotFixSurround.draw()
        # draw fixation point
        dotFix.draw()

        message.setText(clock.getTime())
        message.draw()

        # draw frame
        myWin.flip()

        # handle key presses each frame
        for key in event.getKeys():
            if key in ['escape', 'q']:
                logging.data(msg='User pressed quit')
                myWin.close()
                core.quit()
            elif key[0] in ['5']:
                logging.data(msg='Scanner trigger')
                TriggerPressedArray = np.append(TriggerPressedArray,
                                                clock.getTime())
            elif key in ['1']:
                logging.data(msg='Key1 pressed')
                TargetPressedArray = np.append(TargetPressedArray,
                                               clock.getTime())

    i = i+1

myWin.close()
core.quit()
