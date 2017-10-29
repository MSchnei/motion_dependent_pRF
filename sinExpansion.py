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
    winType='pyglet',  # winType : None, ‘pyglet’, ‘pygame’
    allowGUI=False,
    allowStencil=True,
    fullscr=False,  # for psychoph lab: fullscr = True
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
Conditions = np.array([[0, 11, 22, 33, 44, 55, 63, 4, 10, 40],
                       [2, 0, 1, 2, 0, 1, 2, 2, 2, 2]]).T
Conditions = Conditions.astype(int)

# get timings for the targets
targets = np.array([2, 4, 6, 8, 10])

ExpectedTR = 2

# create array to log key pressed events
TriggerPressedArray = np.array([])
TargetPressedArray = np.array([])

# %%
"""TEXTURE AND MASKS"""

# define the texture
dim = 512
nFrames = 120

x, y = np.meshgrid(np.linspace(-fieldSizeinDeg/2., fieldSizeinDeg/2., dim),
                   np.linspace(-fieldSizeinDeg/2., fieldSizeinDeg/2., dim))


def cart2pol(x, y):
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y, x)
    return t, r


# if necessary scale the vector length
theta, radius = cart2pol(x, y)

phase = np.linspace(0., 4.*np.pi, nFrames)

spatFreq = 1
angularCycles = 12

stimTexture = np.zeros((dim, dim, nFrames))

for ind, t in enumerate(phase):
    ima = np.sin((fieldSizeinDeg/2.) * spatFreq * radius - t)
    stimTexture[..., ind] = ima
    
ctrlTexture = np.zeros((dim, dim, 2))
# get the array that divides field in angular cycles
polCycles = np.sin(angularCycles*theta)
polCycles[np.greater_equal(polCycles, 0)] = 1
polCycles[np.less(polCycles, 0)] = -1
# get radial sine wave gratings
ima = np.sin((fieldSizeinDeg/2.) * spatFreq * radius)
ima = ima * polCycles
ctrlTexture[..., 0] = np.copy(ima)
ctrlTexture[..., 1] = np.copy(ima) * -1


# retrieve the different masks
binMasks = np.load("/Users/Marian/gdrive/Research/MotionLocaliser/" +
                   "RampedMasks.npy")


# %%
"""STIMULI"""
# main stimulus
movRTP = visual.GratingStim(
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
    name='movingRTP',
    autoLog=None,
    autoDraw=False,
    maskParams=None)

# fixation dot
dotFix = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFix',
    radius=2,
    fillColor=[1.0, 0.0, 0.0],
    lineColor=[1.0, 0.0, 0.0],)

# surround of the fixation dot
dotFixSurround = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFixSurround',
    radius=7,
    fillColor=[0.5, 0.5, 0.0],
    lineColor=[0.0, 0.0, 0.0],)

# fixation grid circle
Circle = visual.Polygon(
    win=myWin,
    name='Circle',
    edges=90,
    ori=0,
    units='deg',
    pos=[0, 0],
    lineWidth=2,
    lineColor=[1.0, 1.0, 1.0],
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
    lineColor=[1.0, 1.0, 1.0],
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
    text='Experiment will start soon. \n Waiting for scanner',)

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
nrOfVols = 8
durations = np.ones(nrOfVols)*2
totalTime = ExpectedTR*nrOfVols


def time2frame(t, frameRate=60):
    """Convert time to frames"""
    # time wil be between 0 and TR
    # frames should be between 0 and TR*frameRate
    return t*frameRate


# create function to time ramped onsets and offsets
def raisedCos(steps, T=0.5, beta=0.5):
    """"Create binary wedge-and-ring mask.
    Parameters
    ----------
    steps : float
        Number of points in the output window
    T: float
        The symbol-period
    beta : float
        Roll-off factor
    Returns
    -------
    hf : 1d np.array
        Raised-cosine filter in frequency space
    """

    frequencies = np.linspace(-1/T, 1/T, steps)
    hf = np.empty(len(frequencies))
    for ind, f in enumerate(frequencies):
        if np.less_equal(np.abs(f), (1-beta)/(2*T)):
            hf[ind] = 1
        elif np.logical_and(np.less_equal(np.abs(f), (1+beta)/(2*T)),
                            np.greater(np.abs(f), (1-beta)/(2*T))):
            hf[ind] = 0.5*(1+np.cos((np.pi*T/2)*(np.abs(f)-(1-beta)/2*T)))
        else:
            hf[ind] = 0
    return hf

#window2 = signal.hann(20)[:20/2.]
#
#window = raisedCos(nFrames, T=1, beta=0.3)[:nFrames/2.]
#tempArray = np.zeros(nFrames)
#tempArray[nFrames/4.:-nFrames/4.] = 1
#tempArray[:nFrames/4.] = window
#tempArray[-nFrames/4.:] = window[::-1]

tempArray = np.hstack((np.ones(nFrames/10.), np.zeros(nFrames/10.),
                       np.ones(nFrames/10.), np.zeros(nFrames/10.),
                       np.ones(nFrames/10.), np.zeros(nFrames/2.)))
# get the array to cycle opcaicty
cycOpa = np.nditer(tempArray)

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

    # reset iterator for opacity
    print "refresh"
    cycOpa.reset()

    # get key for motion direction (expanding/ dilation/static)
    keyMotDir = Conditions[i, 1]

    # expanding motion
    if keyMotDir == 0:
        tempIt = np.nditer(np.tile(np.arange(nFrames), 2))
        visTexture = stimTexture

    # contracting motion
    elif keyMotDir == 1:
        tempIt = np.nditer(np.tile(np.arange(nFrames), 2)[::-1])
        visTexture = stimTexture


    # static/flicker control
    elif keyMotDir == 2:
        # define cycle for control condition
#        controlArray = np.sin(phase)
#        controlArray[np.less_equal(np.abs(controlArray), 0.7)] = 0
#        controlArray[np.less(controlArray, 0.)] = -1
#        controlArray[np.greater(controlArray, 0.)] = 1
#        tempIt = np.nditer(controlArray)

        tempIt = np.nditer(np.hstack([np.zeros(nFrames),
                                      np.ones(nFrames)]))
        visTexture = ctrlTexture


    # get key for mask
    keyMask = Conditions[i, 0]
    # get mask
    tmask = np.squeeze(binMasks[:, :, keyMask])

    while clock.getTime() < np.sum(durations[0:i+1]):
        # draw fixation grid (circles and lines)
#        Circle.setSize((2, 2))
#        Circle.draw()
#        Circle.setSize((4, 4))
#        Circle.draw()
#        Circle.setSize((6, 6))
#        Circle.draw()
#        Circle.setSize((8, 8))
#        Circle.draw()
#        Circle.setSize((10, 10))
#        Circle.draw()
#        Line.setOri(0)
#        Line.draw()
#        Line.setOri(45)
#        Line.draw()
#        Line.setOri(90)
#        Line.draw()
#        Line.setOri(135)
#        Line.draw()

        # set texture
        movRTP.tex = visTexture[..., float(tempIt.next())]
        # set opacity such that it follows a raised cosine fashion
        ali = np.copy(float(cycOpa.next()))
        movRTP.opacity = float(ali)
        print float(ali)
        # set mask
        movRTP.mask = tmask
        # draw stimulus
        movRTP.draw()

        # decide whether to draw target
        # first time in target interval? reset target counter to 0!
        if sum(clock.getTime() >= targets) + sum(clock.getTime() < targets + 0.3) == len(targets)+1:
            # display target!
            # change color fix dot surround to red
            dotFixSurround.fillColor = [0.5, 0.0, 0.0]
            dotFixSurround.lineColor = [0.5, 0.0, 0.0]
        else:
            # dont display target!
            # keep color fix dot surround yellow
            dotFixSurround.fillColor = [0.5, 0.5, 0.0]
            dotFixSurround.lineColor = [0.5, 0.5, 0.0]

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
