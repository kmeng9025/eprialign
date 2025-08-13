function varargout = SNRcheckGUI(varargin)
% SNRCHECKGUI MATLAB code for SNRcheckGUI.fig
%      SNRCHECKGUI, by itself, creates a new SNRCHECKGUI or raises the existing
%      singleton*.
%
%      H = SNRCHECKGUI returns the handle to a new SNRCHECKGUI or the handle to
%      the existing singleton*.
%
%      SNRCHECKGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SNRCHECKGUI.M with the given input arguments.
%
%      SNRCHECKGUI('Property','Value',...) creates a new SNRCHECKGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SNRcheckGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SNRcheckGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SNRcheckGUI

% Last Modified by GUIDE v2.5 16-May-2019 12:38:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SNRcheckGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @SNRcheckGUI_OutputFcn, ...
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


% --- Executes just before SNRcheckGUI is made visible.
function SNRcheckGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SNRcheckGUI (see VARARGIN)

% Choose default command line output for SNRcheckGUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SNRcheckGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SNRcheckGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% ----------------------------------------------------------------------
function slider1_Callback(hObject, eventdata, handles)
DrawIt(handles);

% ----------------------------------------------------------------------
function slider1_CreateFcn(hObject, eventdata, handles)

% ----------------------------------------------------------------------
function pushbutton1_Callback(hObject, eventdata, handles)

pathname = safeget(handles, 'pathname', '');
[fname, pathname] = uigetfile({'*.exp;*.tdms'}, 'Select a file', pathname);
if ~isequal(fname,0) && ~isequal(pathname,0)
  set(handles.figure1, 'Name', fname)
  filename = fullfile(pathname, fname);
  
%   [handles.ax, handles.y, handles.dsc] = kv_d01read(filename);
  
  [handles.ax, handles.y, handles.dsc] = kv_smtdmsread(filename);

  handles.n = size(handles.y, 2);
  handles.pathname = pathname;
  guidata(hObject, handles);
  
  n = max(handles.n, 1);
  set(handles.slider1, 'SliderStep', [1/(n-1), 1/(n-1)])
  DrawIt(handles);
end

% ----------------------------------------------------------------------
function DrawIt(handles)

bl_idx = handles.n;
data_idx = bl_idx-1;
idx = floor(get(handles.slider1, 'Value')*(handles.n-1) + 1.5);
if idx < bl_idx
  yy = handles.y(:, bl_idx);
else
  yy = zeros(size(handles.y(:, bl_idx)));
end

yyy  = handles.y(:, idx) - yy;
yyyy = handles.y(:, idx) - handles.y(:, data_idx);
data_maxshot = handles.y(:, data_idx) - handles.y(:, bl_idx);

data_idx = handles.ax.x > 0e-6 & handles.ax.x < 7.0e-6;

signal = max(real(data_maxshot(data_idx)));

cla(handles.axes1)
plot(handles.ax.x, real(yyy), 'Parent', handles.axes1); hold on
plot(handles.ax.x, imag(yyy), 'Parent', handles.axes1);

shots = [1,10,100,1000,10000,10000]*16;

if idx < bl_idx
  plot(handles.ax.x, abs(yyy), 'Parent', handles.axes1);
  plot(handles.ax.x, real(yyyy), 'Parent', handles.axes1);
  legend({'r','i','a','noise'});
  
  noise  = std(real(yyyy(data_idx)));
  set(handles.text2, 'string', sprintf('%i: s=%7.5f n=%7.5f snr=%5.3f', shots(idx), signal, noise, signal/noise));
else
  set(handles.text2, 'string', "baseline"); 
  legend({'r','i'});
end
