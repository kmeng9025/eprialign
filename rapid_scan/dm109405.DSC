#DESC	1.2 * DESCRIPTOR INFORMATION ***********************
*
*	Dataset Type and Format:
*
DSRC	EXP
BSEQ	BIG
IKKF	CPLX
XTYP	IDX
YTYP	NODATA
ZTYP	NODATA
*
*	Item Formats:
*
IRFMT	D
IIFMT	D
*
*	Data Ranges and Resolutions:
*
XPTS	32768
XMIN	0.000000
XWID	131068.000000
*
*	Documentational Text:
*
TITL	'dm109405'
IRNAM	'Intensity'
IINAM	'Intensity'
XNAM	'Time'
IRUNI	''
IIUNI	''
XUNI	'ns'
*
************************************************************
*
#SPL	1.2 * STANDARD PARAMETER LAYER
*
OPER    debbie_mitchell
DATE    01/24/14
TIME    12:51:31
CMNT    
SAMP    
SFOR    
STAG    C
EXPT    CW
OXS1    TADC
AXS1    ETIM
AXS2    NONE
AXS3    
MWFQ    9.630617e+09
MWPW    0.008402
AVGS    29696
*
************************************************************
*
#DSL	1.0 * DEVICE SPECIFIC LAYER
*

.DVC     acqStart, 1.0


.DVC     fieldCtrl, 1.0

AllegroMode        True
Delay              0.0 s
FieldFlyback       On
FieldWait          Wait LED off
GFactor            2.000000
ImgExp             0
ImgFCAddress       3
ImgFCDefField      90.00 G
ImgFCType          Hall
OffResFieldStep    1000.000 G
OffResSetting      Relative Step
OffResStatField    3480.000 G
RSCImagingAmp      True
RSCImgHWMon        0
RSCImgSCFRes       10.0
RSCImgSCWRes       0.0
RSCImgScanFreq     6000.0 Hz
RSCImgScanWidth    3.00 G
RSCImgTrgPhase     25.0
SetToSampleG       False
StaticField        3431.000 G
StaticFieldMon     3479.950 G
SweepDirection     Up

.DVC     freqCounter, 1.0

FrequencyMon       9.630617 GHz
QMonitBridge       On

.DVC     mwBridge, 1.0

AcqFineTuning      Never
AcqScanFTuning     Off
AcqSliceFTuning    Off
BridgeCalib        100.0
Power              8.402 mW
PowerAtten         14 dB
QValue             400

.DVC     recorder, 1.0

BaselineCorr       Off
NbScansAcc         29696
NbScansDone        1
NbScansToDo        1
ReplaceMode        Off
SmoothMode         Manual
SmoothPoints       1

.DVC     transRec, 1.0

AcqMode            Start Dual Trace
BackGrdSubtr       Off
NbCompScansToDo    1
ReplaceMode        Off
TrRecTrgTimeOut    10 s
TransPerScan       29696

*
************************************************************
