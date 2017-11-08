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
strPathParentUp = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
# move to parent_up path
os.chdir(strPathParentUp)

# Name and create specific subject folder
subjFolderName = strPathParentUp + os.path.sep + \
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
    blendMode='avg',
    autoLog=False)

# The size of the field.
fieldSizeinPix = np.round(misc.deg2pix(cfg.fovHeight, moni))

logFile.write('fieldSizeinDeg=' + unicode(cfg.fovHeight) + '\n')
logFile.write('fieldSizeinPix=' + unicode(fieldSizeinPix) + '\n')


# %% CONDITIONS

# get timings for apertures and motion directions
strPathParentUp = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.join(strPathParentUp, 'Conditions',
                        'Conditions_MotDepPrf_run' + str(expInfo['run']) +
                        '.npz')
npzfile = np.load(filename)
conditions = npzfile["conditions"].astype('int')
targetTRs = npzfile["targetTRs"].astype('bool')
targetOffsetSec = npzfile["targetOffsetSec"]
targetDuration = npzfile["targetDuration"]
targetType = npzfile["targetType"]
expectedTR = npzfile["expectedTR"]
targets = np.arange(0, len(conditions)*expectedTR, expectedTR)[targetTRs]
targets = targets + targetOffsetSec
print('TARGETS: ')
print targets

# create array to log key pressed events
triggerPressedArray = np.array([])
targetPressedArray = np.array([])

# log conditions and targets
logFile.write('conditions=' + unicode(conditions) + '\n')
logFile.write('targetOffsetSec=' + unicode(targetOffsetSec) + '\n')
logFile.write('targetDuration=' + unicode(targetDuration) + '\n')
logFile.write('targetType=' + unicode(targetType) + '\n')
logFile.write('expectedTR=' + unicode(expectedTR) + '\n')
logFile.write('targets=' + unicode(targets) + '\n')


# %% TEXTURE AND MASKS

# retrieve the different textures
strPathParentUp = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(strPathParentUp, 'MaskTextures',
                        'Textures_MotDepPrf.npz')
npzfile = np.load(filename)
stimTexture = npzfile["stimTexture"].astype('int8')
ctrlTexture = npzfile["ctrlTexture"].astype('int8')

# retrieve the different masks
filename = os.path.join(strPathParentUp, 'MaskTextures',
                        'Masks_MotDepPrf.npz')
npzfile = np.load(filename)
opaPgDnMasks = npzfile["opaPgDnMasks"].astype('int8')
opaPgUpMasks = npzfile["opaPgUpMasks"].astype('float32')


# %% STIMULI

# main stimulus
radSqrWave = visual.GratingStim(
    myWin,
    tex=np.zeros((cfg.pix/2., cfg.pix/2.)),
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
    autoLog=False,
    autoDraw=False,
    maskParams=None)

radSqrWaveBckgr = visual.GratingStim(
    myWin,
    tex=np.zeros((cfg.pix/2., cfg.pix/2.)),
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
    autoLog=False,
    autoDraw=False,
    maskParams=None)

# fixation dot
dotFix = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFix',
    units='deg',
    radius=0.125,
    fillColor=[0.6, 0.0, 0.0],
    lineColor=[0.6, 0.0, 0.0],)

# surround of the fixation dot
dotFixSurround = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFixSurround',
    units='deg',
    radius=0.19,
    fillColor=[1.0, 1.0, 1.0],
    lineColor=[1.0, 1.0, 1.0],)

# fixation grid line
lineFix = visual.Line(
    win=myWin,
    name='lineFix',
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

# text at start of experiment
triggerText = visual.TextStim(
    win=myWin,
    color='white',
    height=30,
    text='Experiment will start soon.',
    autoLog=False)

# text at end of experiment
targetText = visual.TextStim(
    win=myWin,
    color='white',
    height=30,
    autoLog=False)


# %% TIME AND TIMING PARAMETERS

# get screen refresh rate
refr_rate = myWin.getActualFrameRate()
if refr_rate is not None:
    frameDur = 1.0/round(refr_rate)
else:
    frameDur = 1.0/60.0
print "refr_rate:"
print refr_rate
# log refresh rate and fram duration
logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

# set durations
nrOfVols = len(conditions)
durations = np.ones(nrOfVols)*expectedTR
totalTime = expectedTR*nrOfVols
# log durations
logFile.write('nrOfVols=' + unicode(nrOfVols) + '\n')
logFile.write('durations=' + unicode(durations) + '\n')
logFile.write('totalTime=' + unicode(totalTime) + '\n')

# define opacity on/off cycle in ms
lenCycStim = 400.
lenCycRamp = 50.
lenCycRest = 500.
# log opacity on/off cycle in ms
logFile.write('lenCycStim=' + unicode(lenCycStim) + '\n')
logFile.write('lenCycRamp=' + unicode(lenCycRamp) + '\n')
logFile.write('lenCycRest=' + unicode(lenCycRest) + '\n')
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
cycTransp = np.zeros(2*cfg.nFrames).astype('int8')

# create clock
clock = core.Clock()
logging.setDefaultClock(clock)


# %% FUNCTIONS

def fixationGrid():
    """draw fixation grid (only lines)"""
    lineFix.setOri(0)
    lineFix.draw()
    lineFix.setOri(45)
    lineFix.draw()
    lineFix.setOri(90)
    lineFix.draw()
    lineFix.setOri(135)
    lineFix.draw()

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

    rate = 0

    # get key for masks
    keyMask = conditions[i, 0]
    # get mask to define the opacity values (foreground)
    opaPgUpMask = np.squeeze(opaPgUpMasks[:, :, keyMask])
    # get mask to define the opacity values (background)
    opaPgDnMask = np.squeeze(opaPgDnMasks[:, :, keyMask])

    # set the background mask to opaPgDnMask
    radSqrWaveBckgr.mask = opaPgDnMask
    # set foreground mask to opaPgDnMask
    radSqrWave.mask = opaPgUpMask

    # blank
    if conditions[i, 1] == 0:
        # set timing for the opacity
        visOpa = cycTransp
        # set timing sequence for the texture
        texTime = np.tile(np.repeat(np.array([0, 0]),
                                    cfg.nFrames/(cfg.cycPerSec*2)),
                          cfg.cycPerSec*2).astype('int32')
        # set texture
        visTexture = ctrlTexture

    # static/flicker control
    elif conditions[i, 1] == 1:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime = np.tile(np.repeat(np.array([0, 1]),
                                    cfg.nFrames/(cfg.cycPerSec*2)),
                          cfg.cycPerSec*2).astype('int32')
        # set texture
        visTexture = ctrlTexture

    # contracting motion
    elif conditions[i, 1] == 2:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime = np.tile(np.arange(cfg.nFrames/cfg.cycPerSec),
                          cfg.cycPerSec*2).astype('int32')[::-1]
        # set texture
        visTexture = stimTexture

    # expanding motion
    elif conditions[i, 1] == 3:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime = np.tile(np.arange(cfg.nFrames/cfg.cycPerSec),
                          cfg.cycPerSec*2).astype('int32')
        # set texture
        visTexture = stimTexture

    while clock.getTime() < np.sum(durations[0:i+1]):
        # get interval time
        t = clock.getTime() % expectedTR
        # convert time to respective frame
        frame = t*cfg.nFrames
        # draw fixation grid (circles and lines)
        fixationGrid()

        # set opacity of background aperture
        radSqrWaveBckgr.opacity = visOpa[int(frame)]
        # draw the background aperture
        radSqrWaveBckgr.draw()

        # set the foreground aperture
        radSqrWave.tex = visTexture[..., texTime[int(frame)]]
        # set opacity of foreground aperture
        radSqrWave.opacity = visOpa[int(frame)]
        # draw the foreground aperture
        radSqrWave.draw()

        # decide whether to draw target
        # first time in target interval? reset target counter to 0!
        if (sum(clock.getTime() >= targets) + sum(clock.getTime() <
           targets + targetDuration) == len(targets)+1):
            # display target!
            targetColor = targetType[i]
            # change color fix dot
            if targetColor == 1:
                # change color fix dot surround to brighter red
                dotFix.fillColor = [1.0, 0.0, 0.0]
                dotFix.lineColor = [1.0, 0.0, 0.0]
            elif targetColor == 2:
                # change color fix dot surround to darker red
                dotFix.fillColor = [0.2, 0.0, 0.0]
                dotFix.lineColor = [0.2, 0.0, 0.0]
        else:
            # dont display target!
            # keep color fix dot surround yellow
            dotFix.fillColor = [0.6, 0.0, 0.0]
            dotFix.lineColor = [0.6, 0.0, 0.0]

        # draw fixation point surround
        dotFixSurround.draw()
        # draw fixation point
        dotFix.draw()

        # draw frame
        myWin.flip()
        print "rate:"
        print rate
        rate += 1

        # handle key presses each frame
        for key in event.getKeys():
            if key in ['escape', 'q']:
                logging.data(msg='User pressed quit')
                myWin.close()
                core.quit()
            elif key[0] in ['5']:
                logging.data(msg='Scanner trigger')
                triggerPressedArray = np.append(triggerPressedArray,
                                                clock.getTime())
            elif key in ['1']:
                logging.data(msg='Key1 pressed')
                targetPressedArray = np.append(targetPressedArray,
                                               clock.getTime())
            elif key in ['2']:
                logging.data(msg='Key2 pressed')
                targetPressedArray = np.append(targetPressedArray,
                                               clock.getTime())

    i = i+1

logging.data('EndOfRun' + unicode(expInfo['run']) + '\n')

# %% TARGET DETECTION RESULTS

# calculate target detection results
# create an array 'targetDetected' for showing which targets were detected
targetDetected = np.zeros(len(targets))
if len(targetPressedArray) == 0:
    # if no buttons were pressed
    print "No keys were pressed/registered"
    targetsDet = 0
else:
    # if buttons were pressed:
    for index, target in enumerate(targets):
        for TimeKeyPress in targetPressedArray:
            if (float(TimeKeyPress) >= float(target) and
                    float(TimeKeyPress) <= float(target) + 2):
                targetDetected[index] = 1

logging.data('ArrayOfDetectedTargets' + unicode(targetDetected))
print 'Array Of Detected Targets: ' + str(targetDetected)

# number of detected targets
targetsDet = sum(targetDetected)
logging.data('NumberOfDetectedTargets' + unicode(targetsDet))
# detection ratio
detectRatio = targetsDet/len(targetDetected)
logging.data('RatioOfDetectedTargets' + unicode(detectRatio))

# display target detection results to participant
resultText = 'You detected %i out of %i targets.' % (targetsDet, len(targets))
print resultText
logging.data(resultText)
# also display a motivational slogan
if detectRatio >= 0.95:
    feedbackText = 'Excellent! Keep up the good work'
elif detectRatio < 0.95 and detectRatio > 0.85:
    feedbackText = 'Well done! Keep up the good work'
elif detectRatio < 0.8 and detectRatio > 0.65:
    feedbackText = 'Please try to focus more'
else:
    feedbackText = 'You really need to focus more!'

targetText.setText(resultText+'\n'+feedbackText)
logFile.write(unicode(resultText) + '\n')
logFile.write(unicode(feedbackText) + '\n')
targetText.draw()
myWin.flip()
core.wait(5)


# %% FINISH

# log button presses
logFile.write('triggerPressedArray=' + unicode(triggerPressedArray) + '\n')
logFile.write('targetPressedArray=' + unicode(targetPressedArray) + '\n')
# close qindow
myWin.close()
# quit system
core.quit()
