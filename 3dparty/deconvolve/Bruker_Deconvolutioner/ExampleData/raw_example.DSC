#DESC	1.2 * DESCRIPTOR INFORMATION ***********************
*
*	Dataset Type and Format:
*
DSRC	EXP
BSEQ	BIG
IKKF	REAL
XTYP	IDX
YTYP	IGD
ZTYP	NODATA
*
*	Item Formats:
*
IRFMT	D
YFMT	D
*
*	Data Ranges and Resolutions:
*
XPTS	512
XMIN	386.600000
XWID	4.990234
YPTS	361
YMIN	1.000000
YWID	360.000000
*
*	Documentational Text:
*
TITL	'IMG:1'
IRNAM	'Intensity'
XNAM	'Field'
YNAM	'Angle'
IRUNI	''
XUNI	'G'
YUNI	''
*
************************************************************
*
#SPL	1.2 * STANDARD PARAMETER LAYER
*
OPER    xuser
DATE    09/06/15
TIME    22:15:21
CMNT    
SAMP    
SFOR    
STAG    C
EXPT    CWIMG
OXS1    IADC
AXS1    B0VL
AXS2    GRAD
AXS3    
A1CT    0.03891
A1SW    0.0005
MWFQ    1.107252e+09
IMGP    -90.000
IMGT    -90.000
GRAD    0.500
IMTY    3D
MWPW    0.0034
A2RS    361
A2LL    1
A2SW    1
AVGS    1
SPTP    0.00512
RCAG    60
RCHM    1
B0MA    7.5e-06
B0MF    100000
RCPH    0.0
RCOF    0.0
A1RS    512
RCTC    0
*
************************************************************
*
#DSL	1.0 * DEVICE SPECIFIC LAYER
*

.DVC     acqStart, 1.0


.DVC     fieldCtrl, 1.0

AllegroMode        True
CenterField        389.10 G
Delay              0.0 s
FieldFlyback       On
FieldWait          Wait LED off
GFactor            2.000000
MeasuringHall      False
SetToSampleG       False
StaticFieldMon     386.600 G
SweepDirection     Up
SweepWidth         5.0 G
WidthTM            200.0 G

.DVC     fieldSweep, 1.0


.DVC     freqCounter, 1.0

FrequencyMon       1.107252 GHz
QMonitBridge       On

.DVC     grdUnit, 1.0

AnglePhi           -90.000 deg
AnglePsi           0.5000 G/cm
AngleTheta         -90.000 deg
BandWidth          0.100 G
CurrentCheck       Wait stable
CurrentStrMon      0.500 G/cm
EnableStatic       False
FieldOfView        50.00 mm
FirstAlpha         4.737 deg
ImageOrient        ZX
ImageType          3D
MaxPsi             0.00 deg
MinSW              24.33 G
NPixel             25
NrOfAlpha          19
NrOfBeta           19
NrOfProfs          361
NrOfPsi            1
PixelSize          2.0000 mm
SWTime             2.621 s
SleepSecs          0.0 s
SpecWidth          10.000 G
SpectralRes        0.40000 G/p
TotalTime          30.0 min

.DVC     mwBridge, 1.0

AcqFineTuning      Each Slice Scan
Power              3.400 mW
PowerAtten         20.00 dB

.DVC     ramp2, 1.0

Flyback            False
NbCycles           1
NbPoints           361
StartValue         1
SweepData          
SweepWidth         1
sweepType          Ramp up

.DVC     recorder, 1.0

BaselineCorr       Off
NbScansAcc         1
NbScansDone        1
NbScansToDo        1
ReplaceMode        Off
SmoothMode         Manual
SmoothPoints       1

.DVC     scanEnd, 1.0


.DVC     signalChannel, 1.0

AFCTrap            True
AllegroMode        True
AllowShortCt       False
Calibrated         True
ConvTime           5.12 ms
DModDetectSCT      First
DualDetect         OFF
EliDelay           1.0 us
Enable1stHarm      True
Enable1stHarm90    False
Enable2ndHarm      False
Enable2ndHarm90    False
Enable3rdHarm      False
Enable3rdHarm90    False
Enable4thHarm      False
Enable4thHarm90    False
Enable5thHarm      False
Enable5thHarm90    False
EnableDisp         False
EnableImag         Disable
ExtLockIn          False
ExtTrigger         False
Gain               60 dB
GainB              60 dB
Harmonic           1
HighPass           True
InputPlugA         AC3
InputPlugB         AC3
Integrator         False
IsCalibExp         False
ModAmp             0.075 G
ModFreq            100.00 kHz
ModPhase           0.0
Offset             0.0 %
QuadMode           False
Resolution         512
Resonator          1
SctNorm            True
SctRevision        Allegro
SetAllOrd          False
SetOrdDef          False
SpuExtension       True
SpuRevision        MultiHarmonic
SweepTime          2.62144 s
TimeConst          0
TimeExp            False
TuneCaps           142

*
************************************************************
