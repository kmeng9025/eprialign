function varargout = LoadCalibrationDLG(varargin)
% LOADCALIBRATIONDLG MATLAB code for LoadCalibrationDLG.fig
%      LOADCALIBRATIONDLG, by itself, creates a new LOADCALIBRATIONDLG or raises the existing
%      singleton*.
%
%      H = LOADCALIBRATIONDLG returns the handle to a new LOADCALIBRATIONDLG or the handle to
%      the existing singleton*.
%
%      LOADCALIBRATIONDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LOADCALIBRATIONDLG.M with the given input arguments.
%
%      LOADCALIBRATIONDLG('Property','Value',...) creates a new LOADCALIBRATIONDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LoadCalibrationDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LoadCalibrationDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LoadCalibrationDLG

% Last Modified by GUIDE v2.5 27-Aug-2021 11:19:05

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LoadCalibrationDLG_OpeningFcn, ...
                   'gui_OutputFcn',  @LoadCalibrationDLG_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before LoadCalibrationDLG is made visible.
function LoadCalibrationDLG_OpeningFcn(hObject, ~, handles, varargin)

if isempty(varargin)
  handles.output = CheckPars([]);
  guidata(hObject, handles);
  closereq;
  return;
end

% Choose default command line output for LoadCalibrationDLG
handles.pars = varargin{1};
handles.output = CheckPars(handles.pars);

UpdateControls(handles);

calibrations = epr_spin_probe_calibration;
set(handles.pmCalibrations, 'string', calibrations);

% Update handles structure
guidata(hObject, handles);
uiwait(handles.figure1)

function pars = CheckPars(in_pars)

pars.mG_per_mM = safeget(in_pars, 'mG_per_mM', 2.32);
pars.MDNmG_per_mM = safeget(in_pars, 'MDNmG_per_mM', 0);
pars.LLW_zero_po2 = safeget(in_pars, 'LLW_zero_po2', 12.4);
pars.amp1mM = safeget(in_pars, 'amp1mM', 1);
pars.Q = safeget(in_pars, 'Q', 15);
pars.Qcb = safeget(in_pars, 'Qcb', pars.Q);
pars.Torr_per_mGauss = safeget(in_pars, 'Torr_per_mGauss', 1.84);

function UpdateControls(handles)

handles.epO2.String = num2str(handles.output.Torr_per_mGauss);
handles.eLLW.String = num2str(handles.output.LLW_zero_po2);

handles.e1mM.String = num2str(handles.output.amp1mM);

handles.eAmp.String = num2str(handles.output.mG_per_mM);
handles.eAVERAGEamp.String = num2str(handles.output.MDNmG_per_mM);
handles.eQImage.String = num2str(handles.output.Q);
handles.eQcalibration.String = num2str(handles.output.Qcb);

function handles = ReadControls(handles)

handles.output.Torr_per_mGauss = str2double(handles.epO2.String);
handles.output.LLW_zero_po2 = str2double(handles.eLLW.String);

handles.output.amp1mM = str2double(handles.e1mM.String);

handles.output.mG_per_mM = str2double(handles.eAmp.String);
handles.output.MDNmG_per_mM = str2double(handles.eAVERAGEamp.String);

handles.output.Q = str2double(handles.eQImage.String);
handles.output.Qcb = str2double(handles.eQcalibration.String);

% --- Outputs from this function are returned to the command line.
function varargout = LoadCalibrationDLG_OutputFcn(hObject, ~, handles) 
% Get default command line output from handles structure
if ~isempty(handles)
  varargout{1} = handles.output;
  delete(hObject);
else
  varargout{1} = [];
end

% --- Executes on selection change in pmCalibrations.
function pmCalibrations_Callback(hObject, ~, handles)
probe = handles.pmCalibrations.String{handles.pmCalibrations.Value};
fprintf('Calibration selected: %s\n', probe);
clb = epr_spin_probe_calibration(probe);
disp(clb)

f = 1/pi/2/2.802*1000; % conversion of MS-1 to mG
if contains('IRESE', 'IRESE')
  handles.output.Torr_per_mGauss = clb.slopeT1 / f;
  handles.output.LLW_zero_po2 = clb.interceptT1 * f;
else
  handles.output.Torr_per_mGauss = clb.slopeT2 / f;
  handles.output.LLW_zero_po2 = clb.interceptT2 * f;
end

handles.output = CheckPars(handles.output);
guidata(hObject, handles);
UpdateControls(handles)

% --- Executes on button press in pbOK.
function pbOK_Callback(hObject, ~, handles)
handles = ReadControls(handles);
guidata(hObject, handles);

uiresume

% --- Executes on button press in pbOldValues.
function pbOldValues_Callback(hObject, ~, handles)
handles.output = CheckPars(handles.pars);
guidata(hObject, handles);
UpdateControls(handles)
