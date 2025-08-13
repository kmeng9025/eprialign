#DESC	1.2 * DESCRIPTOR INFORMATION ***********************
*
*	Dataset Type and Format:
*
DSRC	EXP
BSEQ	BIG
IKKF	REAL,REAL,REAL,REAL,REAL,REAL,REAL,REAL,REAL,REAL
XTYP	IDX
YTYP	NODATA
ZTYP	NODATA
*
*	Item Formats:
*
IRFMT	D,D,D,D,D,D,D,D,D,D
*
*	Data Ranges and Resolutions:
*
XPTS	1000
XMIN	3342.500000
XWID	29.570400
*
*	Documentational Text:
*
TITL	'19 SPU BDPA Field Sweep MA=2.0 100kHz allMHs'
IRNAM	'Intensity','','','','','','','','',''
XNAM	'Field'
IRUNI	'','','','','','','','','',''
XUNI	'G'
*
************************************************************
*
#SPL	1.2 * STANDARD PARAMETER LAYER
*
OPER    xuser
DATE    03/04/13
TIME    15:36:33
CMNT    
SAMP    
SFOR    
STAG    C
EXPT    CW
OXS1    IADC
AXS1    B0VL
AXS2    NONE
AXS3    
A1CT    0.33573
A1SW    0.00296
MWFQ    9.410494e+09
MWPW    0.007962
AVGS    1
SPTP    0.04
RCAG    60
RCHM    1
B0MA    0.0002
B0MF    100000
RCPH    0.0
RCOF    0.0
A1RS    1000
RCTC    0
*
************************************************************
*
#DSL	1.0 * DEVICE SPECIFIC LAYER
*

.DVC     acqStart, 1.0


.DVC     fieldCtrl, 1.0

AllegroMode        True
CenterField        3357.30 G
Delay              0.0 s
FieldFlyback       On
FieldWait          Wait LED off
GFactor            2.000000
MeasuringHall      False
SetToSampleG       False
StaticFieldMon     3342.500 G
SweepDirection     Up
SweepWidth         29.6 G
WidthTM            200.0 G

.DVC     fieldSweep, 1.0


.DVC     freqCounter, 1.0

FrequencyMon       9.410494 GHz
QMonitBridge       On

.DVC     mwBridge, 1.0

AcqFineTuning      Never
AcqScanFTuning     Off
AcqSliceFTuning    Off
BridgeCalib        23.4
Power              7.962 mW
PowerAtten         14.0 dB

.DVC     recorder, 1.0

BaselineCorr       Off
NbScansAcc         1
NbScansDone        1
NbScansToDo        1
ReplaceMode        Off
SmoothMode         Manual
SmoothPoints       0

.DVC     scanEnd, 1.0


.DVC     signalChannel, 1.0

AFCTrap            True
AllegroMode        True
AllowShortCt       False
Calibrated         True
ConvFact           9.271e-09
ConvTime           40.00 ms
DModDetectSCT      First
DualDetect         OFF
EliDelay           1.0 us
Enable1stHarm      True
Enable1stHarm90    True
Enable2ndHarm      True
Enable2ndHarm90    True
Enable3rdHarm      True
Enable3rdHarm90    True
Enable4thHarm      True
Enable4thHarm90    True
Enable5thHarm      True
Enable5thHarm90    True
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
ModAmp             2.000 G
ModFreq            100.00 kHz
ModPhase           0.0
Offset             0.0 %
PolyCof            {2;3,9;0} 0,0.99652,0,0,0.00737177,0,0,-0.00559614,0,0,-2.88221e-05,0,0,1.00404e-05,0,0,3.43695e-08,0,0,-5.0404e-09,0,0,-1.4783e-11,0,0,-1.29132e-12,0
QuadMode           False
ResCenter          62.5 mm
ResLength          40 mm
Resolution         1000
Resonator          1
SctNorm            True
SctRevision        Allegro
SetAllOrd          False
SetOrdDef          False
SpuExtension       True
SpuRevision        MultiHarmonic
SweepTime          40.00000 s
TimeConst          0
TimeExp            False
TuneCaps           39

*
************************************************************
