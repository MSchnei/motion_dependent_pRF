# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:50:23 2017

@author: marian
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
import os
import numpy as np
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
from scipy import signal
import config_MotDepPrf as cfg


# %% SAVING and LOGGING

# Store info about experiment and experimental run
expName = 'psychophycis_motDepPrf'  # set experiment name here
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
distanceMon = 58  # [58] in psychoph lab [99] in scanner
widthMon = 53  # [53] in psychoph lab [30] in scanner
PixW = 1920.0  # [1920.0] in psychopy lab [1920.0] in scanner
PixH = 1200.0  # [1080.0] in psychoph lab [1200.0] in scanner

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
    allowStencil=False,
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
                        'Conditions_Psychophysics_run' + str(expInfo['run']) +
                        '.npz')
npzfile = np.load(filename)
conditions = npzfile["conditions"].astype('int8')

# create array to log key pressed events
targetPressedArray = np.array([])

# log conditions and targets
logFile.write('conditions=' + unicode(conditions) + '\n')

# %% TEXTURE AND MASKS

# retrieve the different textures
strPathParentUp = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))

filename = os.path.join(strPathParentUp, 'MaskTextures',
                        'Textures_Psychophysics.npz')
npzfile = np.load(filename)
stimTexture = npzfile["stimTexture"].astype('int8')
ctrlTexture = npzfile["ctrlTexture"].astype('int8')

# retrieve the different masks
filename = os.path.join(strPathParentUp, 'MaskTextures',
                        'Masks_Psychophysics.npz')
npzfile = np.load(filename)
opaPgUpMasks = npzfile["binMasksCos"].astype('float32')

nrOfMasks = len(cfg.ecc)*len(cfg.posShifts)

# %% STIMULI

# main stimulus
radSqrWaveTest = visual.GratingStim(
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
    name='radSqrWaveTest',
    autoLog=False,
    autoDraw=False,
    maskParams=None)

# main stimulus
radSqrWaveSample = visual.GratingStim(
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
    name='radSqrWaveSample',
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
    fillColor=[1.0, -1.0, -1.0],
    lineColor=[1.0, -1.0, -1.0])

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

# log opacity on/off cycle in ms
logFile.write('lenCycInit=' + unicode(cfg.lenCycInit) + '\n')
logFile.write('lenCycStim=' + unicode(cfg.lenCycStim) + '\n')
logFile.write('lenCycRamp=' + unicode(cfg.lenCycRamp) + '\n')
logFile.write('lenCycRest=' + unicode(cfg.lenCycRest) + '\n')
# derive how much of a second the stimlus, blank and ramp period should be
divInit = 1000/cfg.lenCycInit
divStim = 1000/cfg.lenCycStim
divRamp = 1000/cfg.lenCycRamp
divRest = 1000/cfg.lenCycRest
# derive total loop time
loopTime = (4*cfg.lenCycRamp+cfg.lenCycRest+2*cfg.lenCycStim+cfg.lenCycInit
            )/1000.

# define arrays to cycle opacity
cycAlt = np.hstack((
    np.zeros(np.round(cfg.nFrames/divInit)),
    signal.hamming(2*cfg.nFrames/divRamp)[:cfg.nFrames/divRamp],
    np.ones(np.round(cfg.nFrames/divStim)),
    signal.hamming(2*cfg.nFrames/divRamp)[cfg.nFrames/divRamp:],
    np.zeros(np.round(cfg.nFrames/divRest)),
    signal.hamming(2*cfg.nFrames/divRamp)[:cfg.nFrames/divRamp],
    np.ones(np.round(cfg.nFrames/divStim)),
    signal.hamming(2*cfg.nFrames/divRamp)[cfg.nFrames/divRamp:],
    )).astype('float32')
cycTransp = np.zeros(2*cfg.nFrames).astype('int8')

# set timing sequence for the texture
texTimeBlank = np.tile(np.repeat(np.array([0, 0]),
                                 cfg.nFrames/(cfg.cycPerSec*2)),
                       cfg.cycPerSec*2).astype('int8')
texTimeFlicker = np.tile(np.repeat(np.array([0, 1]),
                                   cfg.nFrames/(cfg.cycPerSec*2)),
                         cfg.cycPerSec*2).astype('int8')
texTimeExpd = np.tile(np.arange(cfg.nFrames/cfg.cycPerSec),
                      cfg.cycPerSec*2).astype('int8')
texTimeCntr = np.tile(np.arange(cfg.nFrames/cfg.cycPerSec)[::-1],
                      cfg.cycPerSec*2).astype('int8')

# create clock
clock = core.Clock()
logging.setDefaultClock(clock)
lClock = core.Clock()


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

trials = np.arange(len(conditions))
tloop = True

# Create Counters
trigCount = 0

# give the system time to settle
core.wait(1)

# wait for scanner trigger
triggerText.draw()
myWin.flip()
event.waitKeys(keyList=['5'], timeStamped=False)

# reset clocks
clock.reset()
lClock.reset()
logging.data('StartOfRun' + unicode(expInfo['run']))

while trigCount < len(conditions):

    # set switch for looop to true
    tloop = True
    # reset the clock
    lClock.reset()
    # log the condition
    logging.data(msg=str(conditions[trigCount, :]))

    combiInd = conditions[trigCount, 0]
    posShiftsInd = conditions[trigCount, 1]
    eccInd = conditions[trigCount, 2]

    eccTestMaskNr = np.arange(nrOfMasks
                              ).reshape(-1, 2)[cfg.eccPositions[eccInd],
                                               eccInd]
    eccSampleMaskNr = np.arange(nrOfMasks).reshape(-1, 2)[posShiftsInd, eccInd]

    condTest = cfg.combis[combiInd][0]
    condSample = cfg.combis[combiInd][1]

    # set foreground mask to opaPgDnMask
    radSqrWaveTest.mask = np.squeeze(opaPgUpMasks[:, :, eccTestMaskNr])
    # set the background mask to opaPgDnMask
    radSqrWaveSample.mask = np.squeeze(opaPgUpMasks[:, :, eccSampleMaskNr])

    # blank
    if condSample == 0 and condTest == 0:
        # set timing for the opacity
        visOpa = cycTransp
        # set timing sequence for the texture
        texTime1 = texTimeBlank
        texTime2 = texTimeBlank
        # set texture
        visTexture1 = ctrlTexture
        visTexture2 = ctrlTexture

    # combi1
    elif condSample == 1 and condTest == 2:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeFlicker
        texTime2 = texTimeCntr
        # set texture
        visTexture1 = ctrlTexture
        visTexture2 = stimTexture

    # combi2
    elif condSample == 1 and condTest == 3:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeFlicker
        texTime2 = texTimeExpd
        # set texture
        visTexture1 = ctrlTexture
        visTexture2 = stimTexture

    # combi3
    elif condSample == 2 and condTest == 1:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeCntr
        texTime2 = texTimeFlicker
        # set texture
        visTexture1 = stimTexture
        visTexture2 = ctrlTexture

    # combi4
    elif condSample == 3 and condTest == 1:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeExpd
        texTime2 = texTimeFlicker
        # set texture
        visTexture1 = stimTexture
        visTexture2 = ctrlTexture

    # combi5
    elif condSample == 2 and condTest == 2:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeCntr
        texTime2 = texTimeCntr
        # set texture
        visTexture1 = stimTexture
        visTexture2 = stimTexture

    # combi6
    elif condSample == 3 and condTest == 3:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeExpd
        texTime2 = texTimeExpd
        # set texture
        visTexture1 = stimTexture
        visTexture2 = stimTexture

    # combi7
    elif condSample == 2 and condTest == 3:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeCntr
        texTime2 = texTimeExpd
        # set texture
        visTexture1 = stimTexture
        visTexture2 = stimTexture

    # combi8
    elif condSample == 3 and condTest == 2:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeExpd
        texTime2 = texTimeCntr
        # set texture
        visTexture1 = stimTexture
        visTexture2 = stimTexture

    while tloop:
        # get interval time
        t = lClock.getTime() % loopTime
        # convert time to respective frame
        frame = t*cfg.nFrames
        # draw fixation grid (circles and lines)
        # fixationGrid()

        # blank period
        if lClock.getTime() > 0 and lClock.getTime() <= (cfg.lenCycInit/1000.):
            reponsePeriod = False
            # color of fixation red
            dotFix.fillColor = [1.0, -1.0, -1.0]
            dotFix.lineColor = [1.0, -1.0, -1.0]

        # test stimulus
        elif (lClock.getTime() > (cfg.lenCycInit/1000.) and lClock.getTime() <=
              (cfg.lenCycInit/1000. + 2*cfg.lenCycRamp/1000. +
              cfg.lenCycStim/1000.)):
            reponsePeriod = False

            # set the foreground aperture
            radSqrWaveSample.tex = visTexture1[..., texTime1[int(frame)]]
            # set opacity of background aperture
            radSqrWaveSample.opacity = visOpa[int(frame)]
            # draw the background aperture
            radSqrWaveSample.draw()

            # color of fixation red
            dotFix.fillColor = [1.0, -1.0, -1.0]
            dotFix.lineColor = [1.0, -1.0, -1.0]

        # blank period
        elif (lClock.getTime() > (cfg.lenCycInit/1000. + 2*cfg.lenCycRamp/1000.
              + cfg.lenCycStim/1000.) and lClock.getTime() <=
              (cfg.lenCycInit/1000. + 2*cfg.lenCycRamp/1000.
              + cfg.lenCycStim/1000. + cfg.lenCycRest/1000.)):
            reponsePeriod = False

        # period sample stimulus
        elif (lClock.getTime() > (cfg.lenCycInit/1000. + 2*cfg.lenCycRamp/1000.
              + cfg.lenCycStim/1000. + cfg.lenCycRest/1000.) and
              lClock.getTime() <= loopTime):
            reponsePeriod = False

            # set the foreground aperture
            radSqrWaveTest.tex = visTexture2[..., texTime2[int(frame)]]
            # set opacity of foreground aperture
            radSqrWaveTest.opacity = visOpa[int(frame)]
            # draw the foreground aperture
            radSqrWaveTest.draw()

            # color of fixation red
            dotFix.fillColor = [1.0, -1.0, -1.0]
            dotFix.lineColor = [1.0, -1.0, -1.0]

        # response period
        elif lClock.getTime() > loopTime:
            reponsePeriod = True
            # color of fixation green
            dotFix.fillColor = [-1.0, 1.0, -1.0]
            dotFix.lineColor = [-1.0, 1.0, -1.0]

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

            elif key in ['1']:
                if reponsePeriod:
                    trigCount = trigCount+1
                responsePeriod = False
                tloop = False

            elif key in ['2']:
                if reponsePeriod:
                    trigCount = trigCount+1
                responsePeriod = False
                tloop = False


logging.data('EndOfRun' + unicode(expInfo['run']) + '\n')


# %% FINISH

# log button presses
logFile.write('targetPressedArray=' + unicode(targetPressedArray) + '\n')
# close qindow
myWin.close()
# quit system
core.quit()
