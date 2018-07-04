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
from ctypes import *  # for eyetracker

# %% SET CONTRAST

contrastVal = [0.02, 0.02, 0.02]  # contrast: 5%
# contrastVal = [0.18, 0.18, 0.18]  # contrast: 50%

# %% SAVING and LOGGING

# Store info about experiment and experimental run
expName = 'psychophycis_motDepPrf'  # set experiment name here
expInfo = {
    u'participant': u'pilot',
    u'run': u'01',
    u'flicker': [False, True],
    u'ETused': [False, True]
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
PixH = 1080.0  # [1080.0] in psychoph lab [1200.0] in scanner

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
if expInfo['flicker'] == True:
    # load predefined angular flicker
    ctrlTexture = npzfile["ctrlTexture"].astype('int8')
else:
    # create counter-phase flicker by redefining the control flicker
    ctrlTexture = np.empty((stimTexture.shape[:-1] + (2,)))
    ctrlTexture[..., 0] = np.copy(stimTexture[..., 0])
    ctrlTexture[..., 1] = np.copy(stimTexture[..., 0]) * -1

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
    color=contrastVal,
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
    color=contrastVal,
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

#%%
"""SETUP FOR EYETRACKER"""
# eyetracker set by user?
print(expInfo['ETused'])
if expInfo['ETused']:
    print("Use eyetracker")
    ET = True
else:
    print("Decided not to use eyetracker")
    ET = False

if ET:
    # Eyetracker CONSTANTS (see vpx.h for full listing)
    VPX_STATUS_ViewPointIsRunning = 1
    EYE_A = 0
    VPX_DAT_FRESH = 2
    # load dll
    vpxDllPath = "C:\ViewPoint 2.9.2.5\Interfaces\Windows\ViewPointClient Ethernet Interface\VPX_InterApp.dll"
    # this has to be in same folder as viewPointClient.exe

# CONNECT TO EYETRACKER
if ET:
    #  Load the ViewPoint library
    vpxDll = vpxDllPath
    if (not os.access(vpxDll, os.F_OK)):
        print("WARNING: Invalid vpxDll path; you need to edit the .py file")
        core.quit()
    else:
        print("dll is working")
    cdll.LoadLibrary( vpxDll )
    vpx = CDLL( vpxDll )
    vpx.VPX_SendCommand('say "Hello from MotDepPrf Prep Script" ')
    if (vpx.VPX_GetStatus(VPX_STATUS_ViewPointIsRunning) < 1):
        print("ViewPoint is not running")
        core.quit()


# Define needed structures and and callback function
if ET:
    class RealPoint(Structure):
        pass

    RealPoint._fields_ = [
        ("x", c_float),
        ("y", c_float),
    ]
    # Need to declare a RealPoint variable
    cp = RealPoint(1.1, 1.1)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
# define number of calibration points
NoCP = 12

# define function for eyetracker calibration
if ET:
    def calibrateET(PixW, PixH, NoCP):
        # width and height of window
        WinSize = [PixW, PixH]
        # calibration point size equals 8 pixels
        cpSize = 8
        sd = 5
        # Specify number of Calibration Stimulus Points
        vpx.VPX_SendCommand('calibration_points ' + str(NoCP))
        # calibration of the currently selected point is immediately performed
        vpx.VPX_SendCommand('calibration_snapMode On')
        vpx.VPX_SendCommand('calibration_autoIncrement On')
        vpx.VPX_SendCommand('calibration_PresentationOrder Random')
        # if we would like to restrict the calibration points to a rectangle:
        # coordinates of bounding rectangle, order: Left, Top, Right, Bottom.
        vpx.VPX_SendCommand('calibrationRealRect 0.2 0.3 0.8 0.7')
        # define calibration point
        cpd = visual.Rect(win=myWin, width=cpSize, height=cpSize, units='pix',
                          autoLog=False)
        # green calibration point box
        cpb = visual.Rect(win=myWin, width=cpSize, height=cpSize, units='pix',
                          autoLog=False)
        # white
        cpd.setFillColor([255, 255, 255], u'rgb255')
        # green
        cpb.setFillColor([0, 255, 0], u'rgb255')
        # define calibration clock, will count down
        caliClock = core.CountdownTimer()
        # reset calibration clock to zero
        caliClock.reset()
        # set calibration clock to 1, will count down from 1
        caliClock.add(1.0)
        while caliClock.getTime() > 0:
            pass
        # go trough calibration points
        for p in range(1, NoCP+1):
            # get the coordinates for the first calibration point
            # this line lets ViewPoint return a cp value for every p value
            vpx.VPX_GetCalibrationStimulusPoint(p, byref(cp))
            # the returned value will be in ViewPoint coordinates
            # (0,0 for top left and 1,1 for bottom right)
            # therefore, the values are transformed to python (0,0 is centre)
            # -(1-x)  # calculate position of cp (x, y)
            cpPos = (cp.x*WinSize[0]-(WinSize[0]/2),
                     ((1-cp.y)*WinSize[1])-(WinSize[1]/2))
            cpd.setPos(cpPos, log=False)
            cpb.setPos(cpPos, log=False)

            caliClock.reset()
            # draw calibration point and narrowing box
            for j in range(20, 0, -1):  # go from 20 to 0 in -1 steps
                caliClock.add(0.05)
                # decrease size of cp box with every iteration
                cpb.size = (j/sd*cpSize)
                cpb.draw()
                cpd.draw()
                myWin.flip()
                while caliClock.getTime() > 0.0:
                    pass
            # capture current eye position for current calibration point
            vpx.VPX_SendCommand('calibration_snap ' + str(p))

            for j in range(2, 21):  # go from 2 to 20
                caliClock.add(0.05)
                cpb.size = (j/sd*cpSize)
                cpb.draw()
                cpd.draw()
                myWin.flip()
                while caliClock.getTime() > 0.0:
                    pass

            # handle key presses each frame
            for key in event.getKeys():
                if key in ['escape', 'q']:
                    vpx.VPX_SendCommand('dataFile_Close')
                    myWin.close()
                    core.quit()
            caliClock.add(0.2)
            while caliClock.getTime() > 0.0:
                pass

        # clear the screen
        myWin.flip()

# %%
"""INIT EYETRACKER"""

if ET:
    # open and use new data file and give it a name
    FileName = (expInfo['expName']+'_'+str(expInfo['participant'])+'_'+'Run' +
                '_'+str(expInfo['run'])+'_'+str(expInfo['date'])+'.txt')
    vpx.VPX_SendCommand('dataFile_NewName ' + FileName)

# %% RENDER_LOOP
if ET:
    calibrateET(PixW, PixH, NoCP)

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

    if ET:
        # insert Start marker S into datafile to mark start of condition
        vpx.VPX_SendCommand('dataFile_InsertMarker S')
        BlockText = 'StartOfCondition'+str(conditions[trigCount])
        vpx.VPX_SendCommand('dataFile_InsertString ' + BlockText)

    # set switch for looop to true
    tloop = True
    # reset the clock
    lClock.reset()
    # log the condition
    logging.data(msg=str(conditions[trigCount, :]))

    combiInd = conditions[trigCount, 0]
    posShiftsInd = conditions[trigCount, 1]
    eccInd = conditions[trigCount, 2]

    eccTestMaskNr = np.arange(nrOfMasks).reshape(-1,
        len(cfg.eccPositions))[cfg.eccPositions[eccInd], eccInd]
    eccSampleMaskNr = np.arange(nrOfMasks).reshape(-1,
        len(cfg.eccPositions))[posShiftsInd, eccInd]

    condTest = cfg.combis[combiInd][0]
    condSample = cfg.combis[combiInd][1]

    # set foreground mask to opaPgDnMask
    radSqrWaveTest.mask = np.squeeze(opaPgUpMasks[:, :, eccTestMaskNr])
    # set the background mask to opaPgDnMask
    radSqrWaveSample.mask = np.squeeze(opaPgUpMasks[:, :, eccSampleMaskNr])

    # blank
    if condTest == 0 and condSample == 0:
        # set timing for the opacity
        visOpa = cycTransp
        # set timing sequence for the texture
        texTime1 = texTimeBlank
        texTime2 = texTimeBlank
        # set texture
        visTexture1 = ctrlTexture
        visTexture2 = ctrlTexture

    # combi 1 and 2
    elif condTest == 1 and condSample == 1:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeFlicker
        texTime2 = texTimeFlicker
        # set texture
        visTexture1 = ctrlTexture
        visTexture2 = ctrlTexture

    # combi 3
    elif condTest == 3 and condSample == 1:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeExpd
        texTime2 = texTimeFlicker
        # set texture
        visTexture1 = stimTexture
        visTexture2 = ctrlTexture

    # combi 4
    elif condTest == 1 and condSample == 3:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeFlicker
        texTime2 = texTimeExpd
        # set texture
        visTexture1 = ctrlTexture
        visTexture2 = stimTexture

    # combi 5
    elif condTest == 2 and condSample == 1:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeCntr
        texTime2 = texTimeFlicker
        # set texture
        visTexture1 = stimTexture
        visTexture2 = ctrlTexture

    # combi 6
    elif condTest == 1 and condSample == 2:
        # set timing for the opacity
        visOpa = cycAlt
        # set timing sequence for the texture
        texTime1 = texTimeFlicker
        texTime2 = texTimeCntr
        # set texture
        visTexture1 = ctrlTexture
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
            radSqrWaveTest.tex = visTexture1[..., texTime1[int(frame)]]
            # set opacity of background aperture
            radSqrWaveTest.opacity = visOpa[int(frame)]
            # draw the background aperture
            radSqrWaveTest.draw()

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
            radSqrWaveSample.tex = visTexture2[..., texTime2[int(frame)]]
            # set opacity of foreground aperture
            radSqrWaveSample.opacity = visOpa[int(frame)]
            # draw the foreground aperture
            radSqrWaveSample.draw()

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
                if ET:
                    vpx.VPX_SendCommand('dataFile_Close')
                myWin.close()
                core.quit()

            elif key[0] in ['5']:
                logging.data(msg='Scanner trigger')

            elif key in ['1']:
                if reponsePeriod:
                    trigCount = trigCount+1
                responsePeriod = False
                if ET:
                    vpx.VPX_SendCommand('dataFile_InsertMarker ' + '1')
                    vpx.VPX_SendCommand('dataFile_InsertString ' + 'Key1')
                tloop = False

            elif key in ['2']:
                if reponsePeriod:
                    trigCount = trigCount+1
                responsePeriod = False
                if ET:
                    vpx.VPX_SendCommand('dataFile_InsertMarker ' + '2')
                    vpx.VPX_SendCommand('dataFile_InsertString ' + 'Key2')
                tloop = False


logging.data('EndOfRun' + unicode(expInfo['run']) + '\n')


# %% FINISH
# close the eyetracker data file
if ET:
    vpx.VPX_SendCommand('dataFile_Close')
# log button presses
logFile.write('targetPressedArray=' + unicode(targetPressedArray) + '\n')
# close qindow
myWin.close()
# quit system
core.quit()
