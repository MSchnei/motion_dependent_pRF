# -*- coding: utf-8 -*-
"""
Stimulus presentation for pRF mapping.

The purpose of this script is to present retinotopic mapping stimuli using
Psychopy.
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
import numpy as np
import os

# %%
""" SAVING and LOGGING """
# Store info about experiment and experimental run
expName = 'pRF_mapping_log'  # set experiment name here
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
    'Log_%s' % (expInfo['participant'])
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
logFileName = logFolderName + os.path.sep + '%s_%s_Run_%s_%s' % (
    expInfo['participant'], expInfo['expName'],
    expInfo['run'], expInfo['date'])

# Name and create specific folder for pickle output
outFolderName = dataFolderName + os.path.sep + 'Output'
if not os.path.isdir(outFolderName):
    os.makedirs(outFolderName)
outFileName = outFolderName + os.path.sep + '%s_%s_Run_%s_%s' % (
    expInfo['participant'], expInfo['expName'],
    expInfo['run'], expInfo['date'])

# save a log file and set level for msg to be received
logFile = logging.LogFile(logFileName+'.log', level=logging.INFO)
logging.console.setLevel(logging.WARNING)  # set console to receive warnings


#  %%
"""MONITOR AND WINDOW"""
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
# for psychoph lab: make 'fullscr = True', set size =(1920, 1080)
myWin = visual.Window(
    size=(PixW, PixH),
    screen=0,
    winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
    allowGUI=False,
    allowStencil=True,
    fullscr=True,  # for psychoph lab: fullscr = True
    monitor=moni,
    color=[0, 0, 0],
    colorSpace='rgb',
    units='pix',
    blendMode='avg')

# The size of the field.
fieldSizeinDeg = 24
fieldSizeinPix = np.round(misc.deg2pix(fieldSizeinDeg, moni))

# %%
"""CONDITIONS"""

Conditions = np.array(
    [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
     [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]).T
Conditions = Conditions.astype(int)
# get timings for the targets
TargetOnsetinSec = np.array([2, 4, 6, 8, 10])
# set expected TR
ExpectedTR = 2

# create array to log key pressed events
TriggerPressedArray = np.array([])
TargetPressedArray = np.array([])

# create Positions for the bars
steps = 43
barSize = 3
Positions = np.linspace(0, fieldSizeinDeg-barSize, steps)
Positions = misc.deg2pix(Positions, moni)

#Conditions
#TargetOnsetinSec
#TargetDur
#ExpectedTR
#NrOfSteps = 
#NrOfVols = 


logFile.write('Conditions=' + unicode(Conditions) + '\n')
logFile.write('TargetOnsetinSec=' + unicode(TargetOnsetinSec) + '\n')
#logFile.write('TargetDur=' + unicode(TargetDur) + '\n')


# %%
"""TEXTURE AND MASKS"""

# define the texture
dim = 1024
nFrames = 60

# retrieve the different textures
npzfile = np.load("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/textures.npz")
npzfile.files
horiBar = npzfile["horiBar"]
vertiBar = npzfile["vertiBar"]
wedge = npzfile["wedge"]

# retrieve the different masks
npzfile = np.load("/home/marian/Documents/Testing/CircleBarApertures/carrierPattern/masks.npz")
npzfile.files
horiBarMask = npzfile["horiBarMask"]
vertiBarMask = npzfile["vertiBarMask"]
wedgeMasks = npzfile["wedgeMasks"]


# %%
"""STIMULI"""

# INITIALISE SOME STIMULI
grating = visual.GratingStim(
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
    name='grating',
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
"""FUNCTIONS"""
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


def fixationDotSurround():
    """draw fixation dot surround"""
    dotFixSurround.radius = 0.4
    dotFixSurround.fillColor = [0.0, 0.0, 0.0]
    dotFixSurround.lineColor = [0.0, 0.0, 0.0]
    dotFixSurround.draw()

    dotFixSurround.radius = 0.19
    dotFixSurround.fillColor = [1.0, 1.0, 1.0]
    dotFixSurround.lineColor = [1.0, 1.0, 1.0]
    dotFixSurround.draw()


# %%
"""TIME AND TIMING PARAMETERS"""

# get screen refresh rate
refr_rate = myWin.getActualFrameRate()  # get screen refresh rate
if refr_rate is not None:
    frameDur = 1.0/round(refr_rate)
else:
    frameDur = 1.0/60.0  # couldn't get a reliable measure so guess
logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

# set durations
nrOfVols = len(Conditions)
durations = np.ones(nrOfVols)*2
totalTime = ExpectedTR*nrOfVols

tempIt = np.tile(np.arange(nFrames), 2).astype('int32')

# create clock and Landolt clock
clock = core.Clock()
logging.setDefaultClock(clock)


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

    key = Conditions[i, 0]

    # static/flicker control
    if Conditions[i, 1] == 1:
        grating.pos = (0, Positions[key]),
        visTexture = horiBar
        grating.mask = horiBarMask

    # static/flicker control
    elif Conditions[i, 1] == 2:
        grating.pos = (Positions[key], 0),
        visTexture = vertiBar
        grating.mask = vertiBarMask

    # static/flicker control
    elif Conditions[i, 1] == 3:
        grating.pos = (0, 0),
        visTexture = wedge
        grating.mask = wedgeMasks[..., key]

    while clock.getTime() < np.sum(durations[0:i+1]):

        # get interval time
        t = clock.getTime() % ExpectedTR
        # get respective frame
        frame = t*nFrames
        # draw fixation grid (circles and lines)
        fixationGrid()

        # set the foreground aperture
        grating.tex = visTexture[..., tempIt[int(frame)]]

        grating.draw()

        # draw fixation point surround
        fixationDotSurround()

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

    i = i + 1

# %%
"""CLOSE DISPLAY"""
myWin.close()

# %%
"""FINISH"""
core.quit()
