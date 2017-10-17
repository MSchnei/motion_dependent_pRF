# -*- coding: utf-8 -*-
"""
Motion Localiser according to Huk

@author: Marian Schneider
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, event, core,  monitors, logging, gui, data, misc
from psychopy.tools.coordinatetools import pol2cart, cart2pol
import numpy as np
import os


# %%
""" GENERAL PARAMETERS"""

# set target length
targetDur = 0.3  # in s
# set expected TR
expectedTR = 2  # in s

# set properties for moving dots
# The number of dots
nDots = 600
# specify speed in units per frame
dotSpeed = 8  # deg per s [8]
# dot Life, how long should a dot life
dotLife = 10  # number of frames [10]
# The size of the dots [diameter]
dotSize = 0.2  # in deg
# misc.deg2pix(0.2, myWin.monitor)
# The size of the field.
FieldSizeRadius = 5  # radius in deg
# How close can dots come to center of screen (i.e. fixation cross)
innerBorder = 0.5  # distance from screen center in deg

# set background color
backgrColor = [0.0, 0.0, 0.0]
# set dot color
dotColor = [1, 1, 1]  # from -1 (black) to 1 (white)


# %%
""" SAVING and LOGGING """
# Store info about experiment and experimental run
expName = 'locMt_Huk'  # set experiment name here
expInfo = {
    u'run': u'01',
    u'participant': u'pilot',
    }

# Create GUI at the beginning of exp to get more expInfo
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# get current path and save to variable _thisDir
_thisDir = os.path.dirname(os.path.abspath(__file__))
# get parent path and move up one directory
str_path_parent_up = os.path.abspath(
    os.path.join(os.path.dirname( __file__ ), '..'))

# Name and create specific subject folder
subjFolderName = str_path_parent_up + os.path.sep + '%s_SubjData' % (
    expInfo['participant'])
if not os.path.isdir(subjFolderName):
    os.makedirs(subjFolderName)
# Name and create specific folder for logging results
logFolderName = subjFolderName + os.path.sep + 'Logging'
if not os.path.isdir(logFolderName):
    os.makedirs(logFolderName)
logFileName = logFolderName + os.path.sep + '%s_%s_Run%s_%s' % (
    expInfo['participant'], expInfo['expName'], expInfo['run'],
    expInfo['date'])
# Name and create specific folder for pickle output
outFolderName = subjFolderName + os.path.sep + 'Pickle'
if not os.path.isdir(outFolderName):
    os.makedirs(outFolderName)
outFileName = outFolderName + os.path.sep + '%s_%s_Run%s_%s' % (
    expInfo['participant'], expInfo['expName'], expInfo['run'],
    expInfo['date'])
# Name and create specific folder for BV protocol files
prtFolderName = subjFolderName + os.path.sep + 'Protocols'
if not os.path.isdir(prtFolderName):
    os.makedirs(prtFolderName)

# save a log file and set level for msg to be received
logFile = logging.LogFile(logFileName+'.log', level=logging.INFO)
logging.console.setLevel(logging.WARNING)  # console receives warnings/errors

# create array to log key pressed events
TimeKeyPressedArray = np.array([], dtype=float)

# %%
"""MONITOR AND WINDOW"""
# set monitor information:
distanceMon = 99  # [99] in scanner
widthMon = 30  # [30] in scanner
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
    fullscr=True,  # for psychoph lab: fullscr = True
    monitor=moni,
    color=backgrColor,
    colorSpace='rgb',
    units='deg',
    blendMode='avg'
    )

# %%
"""CONDITIONS AND DURATIONS"""

## Load enpz file containing conditions, durations, and targets
#filename = os.path.join(str_path_parent_up, 'Conditions',
#                        'MtLoc_Huk_run' + str(expInfo['run']) + '.npz')
#npzfile = np.load(filename)
#
## load conditions
#conditions = npzfile['conditions']
#logFile.write('Conditions=' + unicode(conditions) + '\n')
#
## load durations of stimulus and rest
#durations = npzfile['durations']
#logFile.write('Durations=' + unicode(durations) + '\n')
#
## load the target onsets
#targets = npzfile['targets']
#logFile.write('Targets=' + unicode(targets) + '\n')

conditions = np.array([-1, 0, 1, 2, 1, 2, 1, 2, 0, -1])

durations = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

targets = np.array([5, 10])


# %%
"""TIME, TIMING AND CLOCKS"""
# parameters
totalTime = np.sum(durations)
logFile.write('TotalTime=' + unicode(totalTime) + '\n')

# give system time to settle before it checks screen refresh rate
core.wait(1)

# get screen refresh rate
refr_rate = myWin.getActualFrameRate()

if refr_rate is not None:
    print 'refresh rate: %i' % refr_rate
    frameDur = 1.0 / round(refr_rate)
    print 'actual frame dur: %f' % frameDur
else:
    # couldnt get reliable measure, guess
    frameDur = 1.0 / 60.0
    print 'fake frame dur: %f' % frameDur

logFile.write('RefreshRate=' + unicode(refr_rate) + '\n')
logFile.write('FrameDuration=' + unicode(frameDur) + '\n')

# define clock
clock = core.Clock()
logging.setDefaultClock(clock)


# %%
"""STIMULI"""
# divide the dotSpeed by the refresh rate to see how many units (deg) the dot
# travels per frame, not per second
dotSpeed = dotSpeed / refr_rate

# log stimulus properties defined above
logFile.write('nDots=' + unicode(nDots) + '\n')
logFile.write('speed=' + unicode(dotSpeed) + '\n')
logFile.write('dotSize=' + unicode(dotSize) + '\n')
logFile.write('fieldSizeRadius=' + unicode(FieldSizeRadius) + '\n')


def makeLoG(size, sigma, pixel):
    "Function that creates a Laplacian of Gaussian"
    x, y = np.meshgrid(np.linspace(-pixel, pixel, size),
                       np.linspace(-pixel, pixel, size))
    part1 = (1/np.square(sigma))
    part2 = (1-((np.square(x) + np.square(y))/np.square(sigma)))
    part3 = np.exp(-(np.square(x) + np.square(y))/(2*np.square(sigma)))
    return part1 * part2 * part3

# create texture as a Laplacian of Gaussian (LoG)
size = 64
sigma = 0.05
texture = makeLoG(size, sigma, dotSize)

# bring texture in range of -1 and 1
texture = texture/np.max(texture)

# invert the contrast for some of the dots
contrasts = np.ones(int(nDots))
contrasts[np.random.choice(len(contrasts), len(contrasts)/2.)] = -1

# create circular mask for dots
mask = np.ones((size, size))
x, y = np.meshgrid(np.arange(-size/2., size/2.)+0.5,
                   np.arange(-size/2., size/2.)+0.5)
mask[np.greater_equal(np.sqrt(np.square(x) + np.square(y)), size/2.)] = 0

# initialise moving dot stimuli
dotPatch = visual.ElementArrayStim(
    myWin,
    units='deg',
    fieldPos=(0.0, 0.0),
    fieldSize=FieldSizeRadius*2,
    fieldShape='circle',
    nElements=int(nDots),
    sizes=dotSize,
    xys=None,
    colors=(1.0, 1.0, 1.0),
    colorSpace='rgb',
    opacities=1.0,
    depths=0,
    fieldDepth=0,
    oris=0,
    sfs=1/dotSize,
    elementTex=texture,
    elementMask=mask,
    contrs=contrasts,
    phases=0,
    texRes=64,
    interpolate=False,
    name='rdk',
    autoLog=None,
    maskParams=None
    )

dotPatch.setTex(texture)

# fixation dot
dotFix = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFix',
    units='pix',
    radius=5,
    fillColor=[1.0, 0.0, 0.0],
    lineColor=[1.0, 0.0, 0.0],
    )

dotFixSurround = visual.Circle(
    myWin,
    autoLog=False,
    name='dotFix',
    units='pix',
    radius=10,
    fillColor=[0.5, 0.5, 0.0],
    lineColor=[0.0, 0.0, 0.0],
    )

# control text
controlText = visual.TextStim(
    win=myWin,
    colorSpace='rgb',
    color=[1.0, 1.0, 1.0],
    height=0.5,
    pos=(0.0, -4.0),
    autoLog=False,
    )

# text at the beginning of the experiment
triggerText = visual.TextStim(
    win=myWin,
    colorSpace='rgb',
    color=[1.0, 1.0, 1.0],
    height=0.5,
    text='Experiment will start soon. Waiting for scanner'
    )

# %%
"""FUNCTIONS"""

# %% calculate probability for expanding dots

# distance that dots travel from inner border on one frame
Sexp = innerBorder + dotSpeed
# calculate alpha
alphaExp = Sexp/innerBorder
# probability to be repositioned in area A
pAexp = alphaExp*(Sexp**2-innerBorder**2)/(
    (alphaExp-1)*(FieldSizeRadius**2-Sexp**2)+alphaExp*(
        Sexp**2-innerBorder**2))

# %% calculate probability for contracting dots

# distance that dots travel from inner border on one frame
Scontr = FieldSizeRadius - dotSpeed
# calculate alpha
alphaContr = FieldSizeRadius/Scontr
# probability to be repositioned in area A
pAcontr = alphaContr*(FieldSizeRadius**2-Scontr**2)/(
    alphaContr*(FieldSizeRadius**2-Scontr**2) + (alphaContr-1)*(
        Scontr**2-innerBorder**2))

# %% function to determine initial dot positions
def dots_init(nDots):
    # initialise angle for each dot as a uniform distribution
    dotsTheta = np.random.uniform(0, 360, nDots)
    # initialise radius for each dot
    # in order to get an overall uniform distribution, the radius must not be
    # picked from a uniform distribution, but as pdf_r = (2/R^2)*r
    dotsRadius = np.sqrt(
        (np.square(FieldSizeRadius) - np.square(innerBorder)) *
        np.random.rand(nDots) + np.square(innerBorder))
    # convert from polar to Cartesian
    dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)
    # create array frameCount
    frameCount = np.random.uniform(0, dotLife, size=len(dotsX)).astype(int)
    return dotsX, dotsY, frameCount

# %% functions to update dot positions
# function that updates according to the wrap-around procedure described
# in clifford et al.
def dots_update_wrap(dotsX, dotsY, dotSpeed=dotSpeed):
        # convert to polar coordinates
    dotsTheta, dotsRadius = cart2pol(dotsX, dotsY)
    # update radius
    dotsRadius = (dotsRadius+dotSpeed)
    # prepare array for dots that will die from falling out
    lgcOutFieldDots = np.zeros(len(dotsTheta), dtype='bool')

    # decide which dots fall out during expansion
    if dotSpeed > 0:
        # create lgc for elems where radius too large (expansion)
        lgcOutFieldDots = (dotsRadius >= FieldSizeRadius)
        # how many dots should go in area A?
        numDotsAreaA = np.sum(
            np.random.choice(np.arange(2), p=[1-pAexp, pAexp],
                             size=np.sum(lgcOutFieldDots)))
        # get logical for area A
        lgcAdots = np.zeros(len(dotsTheta), dtype='bool')
        lgcAdots[np.where(lgcOutFieldDots)[0][:numDotsAreaA]] = True
        # calculate new radius for dots appearing in region A
        dotsRadius[lgcAdots] = innerBorder*np.sqrt(
            (np.square(alphaExp)-1)*np.random.rand(sum(lgcAdots))+1)
        # get logical for area B
        lgcBdots = np.zeros(len(dotsTheta), dtype='bool')
        lgcBdots[np.where(lgcOutFieldDots)[0][numDotsAreaA:]] = True
        # calculate new radius for dots appearing in region B
        dotsRadius[lgcBdots] = np.sqrt(
            (np.square(FieldSizeRadius) -
             np.square(alphaExp)*np.square(innerBorder)
             )*np.random.rand(sum(lgcBdots)) +
            np.square(alphaExp)*np.square(innerBorder)
            )

    # decide which dots fall out during contraction
    elif dotSpeed < 0:
        # create lgc for elems where radius too small (contraction)
        lgcOutFieldDots = (dotsRadius <= innerBorder)
        # how many dots should go in area A?
        numDotsAreaA = np.sum(
            np.random.choice(np.arange(2), p=[1-pAcontr, pAcontr],
                             size=np.sum(lgcOutFieldDots)))
        # get logical for area A
        lgcAdots = np.zeros(len(dotsTheta), dtype='bool')
        lgcAdots[np.where(lgcOutFieldDots)[0][:numDotsAreaA]] = True
        # calculate new radius for dots appearing in region A
        dotsRadius[lgcAdots] = Scontr*np.sqrt(
            (np.square(alphaContr)-1)*np.random.rand(sum(lgcAdots))+1)
        # get logical for area B
        lgcBdots = np.zeros(len(dotsTheta), dtype='bool')
        lgcBdots[np.where(lgcOutFieldDots)[0][numDotsAreaA:]] = True
        # calculate new radius for dots appearing in region B
        dotsRadius[lgcBdots] = np.sqrt(
            (np.square(Scontr) -
             np.square(innerBorder)
             )*np.random.rand(sum(lgcBdots)) +
            np.square(innerBorder)
            )

    # calculate new angle for all dots that died (from age or falling)
    dotsTheta[lgcOutFieldDots] = np.random.uniform(
        0, 360, sum(lgcOutFieldDots))
    # convert from polar to Cartesian
    dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)

    return dotsX, dotsY


# function that updates according to dot life time
def dots_update_lifetime(dotsX, dotsY, frameCount, dotSpeed=dotSpeed,
                         frameDeathAfter=np.inf):
    # convert to polar coordinates
    dotsTheta, dotsRadius = cart2pol(dotsX, dotsY)
    # update radius
    dotsRadius = (dotsRadius+dotSpeed)
    # update frameCount
    frameCount += 1
    # prepare array for dots that will die from falling out
    lgcOutFieldDots = np.zeros(len(dotsTheta), dtype='bool')

    # decide which dots fall out during expansion
    if dotSpeed > 0:
        # create lgc for elems where radius too large (expansion)
        lgcOutFieldDots = (dotsRadius >= FieldSizeRadius)
    # decide which dots fall out during contraction
    elif dotSpeed < 0:
        # create lgc for elems where radius too small (contraction)
        lgcOutFieldDots = (dotsRadius <= innerBorder)

    # decide which dots will die because they got too old
    lgcFrameDeath = (frameCount >= frameDeathAfter)
    # combine logicals from dots that died due to fell out and high age
    lgcDeath = np.logical_or(lgcOutFieldDots, lgcFrameDeath)

    # calculate new radius for dots that died
    dotsRadius[lgcDeath] = np.sqrt(
        (np.square(FieldSizeRadius) - np.square(innerBorder)) *
        np.random.rand(sum(lgcDeath)) + np.square(innerBorder))
    # calculate new angle for all dots that died (from age or falling)
    dotsTheta[lgcDeath] = np.random.uniform(0, 360, sum(lgcDeath))

    # reset the counter for newborn dots that died of high age
    frameCount[lgcFrameDeath] = 0
    # reset the counter for newborn dots that died from falling out
    frameCount[lgcOutFieldDots] = np.random.uniform(
        0, dotLife, size=sum(lgcOutFieldDots)).astype(int)

    # convert from polar to Cartesian
    dotsX, dotsY = pol2cart(dotsTheta, dotsRadius)

    return dotsX, dotsY, frameCount

# target function
nrOfTargetFrames = int(targetDur/frameDur)
print "number of target frames"
print nrOfTargetFrames

# set initial value for target counter
mtargetCounter = nrOfTargetFrames+1
def target(mtargetCounter):
    t = clock.getTime()
    # first time in target interval? reset target counter to 0!
    if sum(t >= targets) + sum(t < targets + frameDur) == len(targets) + 1:
        mtargetCounter = 0
    # below number of target frames? display target!
    if mtargetCounter < nrOfTargetFrames:
        # change color fix dot surround to red
        dotFixSurround.fillColor = [0.5, 0.0, 0.0]
        dotFixSurround.lineColor = [0.5, 0.0, 0.0]
    # above number of target frames? dont display target!
    else:
        # keep color fix dot surround yellow
        dotFixSurround.fillColor = [0.5, 0.5, 0.0]
        dotFixSurround.lineColor = [0.5, 0.5, 0.0]

    # update mtargetCounter
    mtargetCounter = mtargetCounter + 1

    return mtargetCounter

# %%
"""RENDER_LOOP"""
# Create Counters
i = 0  # counter for blocks
# draw dots for the first time [inward dots]
dotsX, dotsY, frameCntsIn = dots_init(nDots)
# set x and y positions to initialized values
dotPatch.setXYs(np.array([dotsX, dotsY]).transpose())

# define DirTime and calculate FlickerCycle
DirTime = 1  # move in one dir before moving in opposite
AxisTime = DirTime*2  # because we have 2 dir states (back and forth)

# give system time to settle before stimulus presentation
core.wait(1.0)

# wait for scanner trigger
triggerText.draw()
myWin.flip()
event.waitKeys(keyList=['5'], timeStamped=False)
# reset clock
clock.reset()
logging.data('StartOfRun' + unicode(expInfo['run']))

while clock.getTime() < totalTime:
    # low-level rest (only central fixation dot)
    if conditions[i] == -1:
        # set loopDotSpeed to zero
        loopDotSpeed = 0
        # set loopDotLife to inf
        loopDotLife = np.inf
        # set opacaities
        dotPatch.opacities = 0

    # dynamic static dots rest
    elif conditions[i] == 0:
        # set loopDotSpeed to zero
        loopDotSpeed = 0
        # set loopDotLife to inf
        loopDotLife = dotLife
        # set opacity to 1 for all static
        dotPatch.opacities = 1

    # dilating motion
    elif conditions[i] == 1:
        # set loopDotSpeed to dotSpeed
        loopDotSpeed = dotSpeed
        # set loopDotLife to dotLife
        loopDotLife = dotLife  # dotLife
        # set opacaities
        dotPatch.opacities = 1

    # contracting motion
    elif conditions[i] == 2:
        # set loopDotSpeed to dotSpeed
        loopDotSpeed = -dotSpeed
        # set loopDotLife to dotLife
        loopDotLife = dotLife  # dotLife
        # set opacaities
        dotPatch.opacities = 1

    while clock.getTime() < np.sum(durations[0:i+1]):
        # update dots
        t = clock.getTime()

        dotsX, dotsY, frameCntsIn = dots_update_lifetime(
            dotsX, dotsY, frameCntsIn, dotSpeed=loopDotSpeed,
            frameDeathAfter=loopDotLife)

#        dotsX, dotsY = dots_update_wrap(dotsX, dotsY, dotSpeed=loopDotSpeed)

        dotPatch.setXYs(np.array([dotsX, dotsY]).transpose())

        # draw dots
        dotPatch.draw()

        # update target
        mtargetCounter = target(mtargetCounter)

        # draw fixation point surround
        dotFixSurround.draw()

        # draw fixation point
        dotFix.draw()

        # draw control text
        # controlText.setText(clock.getTime())
        # controlText.draw()

        myWin.flip()

        # handle key presses each frame
        for keys in event.getKeys():
            if keys[0] in ['escape', 'q']:
                myWin.close()
                core.quit()
            elif keys in ['1']:
                TimeKeyPressedArray = np.append([TimeKeyPressedArray],
                                                [clock.getTime()])
                logging.data(msg='Key1 pressed')

    # update counter
    i = i + 1

# log end of run
logging.data('EndOfRun' + unicode(expInfo['run']))

# %%
"""TARGET DETECTION RESULTS"""
# calculate target detection results
# create an array 'targetDetected' for showing which targets were detected
targetDetected = np.zeros(len(targets))
if len(TimeKeyPressedArray) == 0:
    # if no buttons were pressed
    print "No keys were pressed/registered"
    targetsDet = 0
else:
    # if buttons were pressed:
    for index, target in enumerate(targets):
        for TimeKeyPress in TimeKeyPressedArray:
            if (float(TimeKeyPress) >= float(target) and
                    float(TimeKeyPress) <= float(target)+2):
                targetDetected[index] = 1

logging.data('ArrayOfDetectedTargets' + unicode(targetDetected))
print 'Array Of Detected Targets:'
print targetDetected

# number of detected targets
targetsDet = sum(targetDetected)
logging.data('NumberOfDetectedTargets' + unicode(targetsDet))
# detection ratio
DetectRatio = targetsDet/len(targetDetected)
logging.data('RatioOfDetectedTargets' + unicode(DetectRatio))

# display target detection results to participant
resultText = 'You have detected %i out of %i targets.' % (targetsDet,
                                                          len(targets))
print resultText
logging.data(resultText)
# also display a motivational slogan
if DetectRatio >= 0.95:
    feedbackText = 'Excellent! Keep up the good work'
elif DetectRatio < 0.95 and DetectRatio >= 0.85:
    feedbackText = 'Well done! Keep up the good work'
elif DetectRatio < 0.85 and DetectRatio >= 0.65:
    feedbackText = 'Please try to focus more'
else:
    feedbackText = 'You really need to focus more!'

targetText = visual.TextStim(
    win=myWin,
    color='white',
    height=0.5,
    pos=(0.0, 0.0),
    autoLog=False,
    )
targetText.setText(resultText+feedbackText)
logFile.write(unicode(resultText) + '\n')
logFile.write(unicode(feedbackText) + '\n')
targetText.draw()
myWin.flip()
core.wait(5)
myWin.close()

# %%
"""SAVE DATA"""
# log important parameters
try:
    logFile.write('TargetDuration=' + unicode(targetDur) + '\n')
    logFile.write('TimeKeyPressedArray=' + unicode(TimeKeyPressedArray) + '\n')
except:
    print '(Important parameters could not be logged.)'

# create a pickle file with important arrays
try:
    os.chdir(outFolderName)
    # create python dictionary containing important arrays
    output = {'ExperimentName': expInfo['expName'],
              'Date': expInfo['date'],
              'SubjectID': expInfo['participant'],
              'Run_Number': expInfo['run'],
              'Conditions': conditions,
              'Durations': durations,
              'KeyPresses': TimeKeyPressedArray,
              'DetectedTargets': targetDetected,
              }
    # save dictionary as a pickle in output folder
    misc.toFile(outFileName + '.pickle', output)
    print 'Pickle data saved as: ' + outFileName + '.pickle'
    print "***"
    os.chdir(_thisDir)
except:
    print '(OUTPUT folder could not be created.)'

# create prt files for BV
try:
    os.chdir(prtFolderName)

    durationsMsec = (durations*1000)
    durationsMsec = durationsMsec.astype(int)

    # Set Conditions Names
    CondNames = ['Fixation',
                 'MoveC',
                 'MoveL',
                 'MoveR',
                 'StaticC',
                 'StaticL',
                 'StaticR',
                 ]

    # Number code the conditions
    from collections import OrderedDict
    stimTypeDict = OrderedDict()
    stimTypeDict[CondNames[0]] = [-1]
    stimTypeDict[CondNames[1]] = [1]
    stimTypeDict[CondNames[2]] = [2]
    stimTypeDict[CondNames[3]] = [3]
    stimTypeDict[CondNames[4]] = [4]
    stimTypeDict[CondNames[5]] = [5]
    stimTypeDict[CondNames[6]] = [6]

    # Color code the conditions
    colourTypeDict = {
        CondNames[0]: '64 64 64',
        CondNames[1]: '255 170 0',
        CondNames[2]: '170 0 0',
        CondNames[3]: '0 170 0',
        CondNames[4]: '255 255 0',
        CondNames[5]: '255 0 0',
        CondNames[6]: '0 255 0',
        }

    # Defining a function will reduce the code length significantly.
    def idxAppend(iteration, enumeration, dictName, outDict):
        if int(enumeration) in range(stimTypeDict[dictName][0],
                                     stimTypeDict[dictName][-1]+1
                                     ):
            outDict = outDict.setdefault(dictName, [])
            outDict.append(iteration)

    # Reorganization of the protocol array (finding and saving the indices)
    outIdxDict = {}  # an empty dictionary

    # Please take a deeper breath.
    for i, j in enumerate(conditions):
        for k in stimTypeDict:  # iterate through each key in dict
            idxAppend(i, j, k, outIdxDict)

    print outIdxDict

    # Creation of the Brainvoyager .prt custom text file
    prtName = '%s_%s_Run%s_%s.prt' % (expInfo['participant'],
                                      expInfo['expName'],
                                      expInfo['run'],
                                      expInfo['date'])

    file = open(prtName, 'w')
    header = ['FileVersion: 2\n',
              'ResolutionOfTime: msec\n',
              'Experiment: %s\n' % expName,
              'BackgroundColor: 0 0 0\n',
              'TextColor: 255 255 202\n',
              'TimeCourseColor: 255 255 255\n',
              'TimeCourseThick: 3\n',
              'ReferenceFuncColor: 192 192 192\n',
              'ReferenceFuncThick: 2\n'
              'NrOfConditions: %s\n' % str(len(stimTypeDict))
              ]

    file.writelines(header)

    # Conditions/predictors
    for i in stimTypeDict:  # iterate through each key in stim. type dict
        h = i

        # Write the condition/predictor name and put the Nr. of repetitions
        file.writelines(['\n',
                         i+'\n',
                         str(len(outIdxDict[i]))
                         ])

        # iterate through each element, define onset and end of each condition
        for j in outIdxDict[i]:
            onset = int(sum(durationsMsec[0:j+1]) - durationsMsec[j] + 1)
            file.write('\n')
            file.write(str(onset))
            file.write(' ')
            file.write(str(onset + durationsMsec[j]-1))
        # contiditon color
        file.write('\nColor: %s\n' % colourTypeDict[h])
    file.close()
    print 'PRT files saved as: ' + prtFolderName + '\\' + prtName
    os.chdir(_thisDir)
except:
    print '(PRT files could not be created.)'

# %%
"""FINISH"""
core.quit()

#numElem = [len(bla) for bla in radi]
#numElem = np.array(numElem)
#numElem = numElem[:200]
#
#items = [item for sublist in radi for item in sublist]
#items = np.array(items)
#np.diff(np.histogram(items)[0])




