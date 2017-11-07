# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:46:04 2017

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import numpy as np
import os
import config_MotDepPrf as cfg
from scipy import signal
from psychopy import visual, event, core,  monitors, logging, gui, data, misc


# %% SAVING and LOGGING

# Store info about experiment and experimental run
expName = 'motDepPrf'  # set experiment name here
expInfo = {
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


# %% MONITOR AND WINDOW

# set monitor information:
distanceMon = 29.5  # [99 for Nova coil]
widthMon = 35  # [30 for Nova coil]
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
fieldSizeinPix = np.round(misc.deg2pix(cfg.fovHeight, moni))

logFile.write('fieldSizeinDeg=' + unicode(cfg.fovHeight) + '\n')
logFile.write('fieldSizeinPix=' + unicode(fieldSizeinPix) + '\n')


# %% CONDITIONS

# get timings for apertures and motion directions
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.join(str_path_parent_up, 'Conditions',
                        'Conditions_MotDepPrf_run' + str(expInfo['run']) +
                        '.npz')
npzfile = np.load(filename)
Conditions = npzfile["Conditions"].astype('int')
TargetTRs = npzfile["TargetTRs"].astype('bool')
TargetOnsetinSec = npzfile["TargetOnsetinSec"]
TargetDuration = npzfile["TargetDuration"]
ExpectedTR = npzfile["ExpectedTR"]
Targets = np.arange(0, len(Conditions)*ExpectedTR, ExpectedTR)[TargetTRs]
Targets = Targets + TargetOnsetinSec
print('TARGETS: ')
print Targets

#Conditions = np.array(
#    [[0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 21, 22, 23, 24, 25, 26, 0, 0, 40, 41, 42,
#      43, 44, 45, 0, 0, 46, 47, 48, 49, 50, 51],
#     [0, 0, 3, 3, 3, 3, 3, 3, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2,
#      0, 0, 3, 3, 3, 3, 3, 3]]).T
#Conditions = Conditions.astype(int)

# create array to log key pressed events
TriggerPressedArray = np.array([])
TargetPressedArray = np.array([])


# %% TEXTURE AND MASKS

# retrieve the different textures
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Textures_MotDepPrf.npz')
npzfile = np.load(filename)

stimTexture = npzfile["stimTexture"].astype('int8')
ctrlTexture = npzfile["ctrlTexture"].astype('int8')

# retrieve the different masks
filename = os.path.join(str_path_parent_up, 'MaskTextures',
                        'Masks_MotDepPrf.npz')
npzfile = np.load(filename)

opaPgDnMasks = npzfile["opaPgDnMasks"].astype('int32')
opaPgUpMasks = npzfile["opaPgUpMasks"].astype('float32')


# %% STIMULI

# main stimulus
radSqrWave = visual.GratingStim(
    myWin,
    tex=np.zeros((cfg.pix, cfg.pix)),
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
    tex=np.zeros((cfg.pix, cfg.pix)),
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
    name='radSqrWaveBckgr',
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
    interpolate=True)

triggerText = visual.TextStim(
    win=myWin,
    color='white',
    height=30,
    text='Experiment will start soon.',)

targetText = visual.TextStim(
    win=myWin,
    color='white',
    height=30,
    autoLog=False)


# %% TIME AND TIMING PARAMETERS

# get screen refresh rate
refr_rate = myWin.getActualFrameRate()  # get screen refresh rate
if refr_rate is not None:
    frameDur = 1.0/round(refr_rate)
else:
    frameDur = 1.0/cfg.nFrames  # couldn't get a reliable measure so guess
print "refr_rate:"
print refr_rate
logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

# set durations
nrOfVols = len(Conditions)
Durations = np.ones(nrOfVols)*ExpectedTR
totalTime = ExpectedTR*nrOfVols

# define on/off cycle in ms
lenCycStim = 350.
lenCycRamp = 75.
lenCycRest = 500.
# derive how much of a second that is
divStim = 1000/lenCycStim
divRamp = 1000/lenCycRamp
divRest = 1000/lenCycRest

# define arrays to cycle opacity
cycAlt = np.hstack((
    signal.hamming(2*cfg.nFrames/divRamp)[:cfg.nFrames/divRamp],
    np.ones(np.round(cfg.nFrames/divStim)),
    signal.hamming(2*cfg.nFrames/divRamp)[cfg.nFrames/divRamp:],
    np.zeros(np.round(cfg.nFrames/divRest)),
    signal.hamming(2*cfg.nFrames/divRamp)[:cfg.nFrames/divRamp],
    np.ones(np.round(cfg.nFrames/divStim)),
    signal.hamming(2*cfg.nFrames/divRamp)[cfg.nFrames/divRamp:],
    np.zeros(np.round(cfg.nFrames/divRest)),
    )).astype('float32')
cycTransp = np.zeros(2*cfg.nFrames).astype('float32')

# create clock
clock = core.Clock()
logging.setDefaultClock(clock)


# %% FUNCTIONS

def fixationGrid():
    """draw fixation grid (circles and lines)"""
    Circle.setSize((2, 2))
    Circle.draw()
    Circle.setSize((5, 5))
    Circle.draw()
    Circle.setSize((10, 10))
    Circle.draw()
    Circle.setSize((20, 20))
    Circle.draw()
    Circle.setSize((30, 30))
    Circle.draw()
    Line.setOri(0)
    Line.draw()
    Line.setOri(45)
    Line.draw()
    Line.setOri(90)
    Line.draw()
    Line.setOri(135)
    Line.draw()

# %% RENDER_LOOP

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

    # blank
    if Conditions[i, 1] == 0:
        visOpa = cycTransp
        tempIt = np.tile(np.repeat(np.array([0, 0]),
                                   cfg.nFrames/(cfg.cycPerSec*2)),
                         cfg.cycPerSec*2).astype('int32')
        visTexture = ctrlTexture

    # static/flicker control
    elif Conditions[i, 1] == 1:
        visOpa = cycAlt
        tempIt = np.tile(np.repeat(np.array([0, 1]),
                                   cfg.nFrames/(cfg.cycPerSec*2)),
                         cfg.cycPerSec*2).astype('int32')
        visTexture = ctrlTexture

    # contracting motion
    elif Conditions[i, 1] == 2:
        visOpa = cycAlt
        tempIt = np.tile(np.arange(cfg.nFrames/cfg.cycPerSec),
                         cfg.cycPerSec*2).astype('int32')[::-1]
        visTexture = stimTexture

    # expanding motion
    elif Conditions[i, 1] == 3:
        visOpa = cycAlt
        tempIt = np.tile(np.arange(cfg.nFrames/cfg.cycPerSec),
                         cfg.cycPerSec*2).astype('int32')
        visTexture = stimTexture

    while clock.getTime() < np.sum(Durations[0:i+1]):
        # get interval time
        t = clock.getTime() % ExpectedTR
        # convert time to respective frame
        frame = t*cfg.nFrames
        # draw fixation grid (circles and lines)
        fixationGrid()

        # set opacity of background aperture
        radSqrWaveBckgr.opacity = visOpa[int(frame)]
        # draw the background aperture
        radSqrWaveBckgr.draw()

        # set the foreground aperture
        radSqrWave.tex = visTexture[..., tempIt[int(frame)]]
        # set foreground mask to opaPgDnMask
        radSqrWave.mask = opaPgUpMask
        # set opacity of foreground aperture
        radSqrWave.opacity = visOpa[int(frame)]
        # draw the foreground aperture
        radSqrWave.draw()

        # decide whether to draw target
        # first time in target interval? reset target counter to 0!
        if (sum(clock.getTime() >= Targets) + sum(clock.getTime() <
           Targets + 0.3) == len(Targets)+1):
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
            elif key in ['2']:
                logging.data(msg='Key2 pressed')
                TargetPressedArray = np.append(TargetPressedArray,
                                               clock.getTime())

    i = i+1


# %% FINISH

# close qindow
myWin.close()
# quit system
core.quit()
