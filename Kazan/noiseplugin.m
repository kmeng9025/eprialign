function varargout = noiseplugin(varargin)
% NOISEPLUGIN MATLAB code for noiseplugin.fig
%      NOISEPLUGIN, by itself, creates a new NOISEPLUGIN or raises the existing
%      singleton*.
%
%      H = NOISEPLUGIN returns the handle to a new NOISEPLUGIN or the handle to
%      the existing singleton*.
%
%      NOISEPLUGIN('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in NOISEPLUGIN.M with the given input arguments.
%
%      NOISEPLUGIN('Property','Value',...) creates a new NOISEPLUGIN or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before noiseplugin_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to noiseplugin_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help noiseplugin

% Last Modified by GUIDE v2.5 10-Mar-2022 10:25:24

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @noiseplugin_OpeningFcn, ...
                   'gui_OutputFcn',  @noiseplugin_OutputFcn, ...
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


% --- Executes just before noiseplugin is made visible.
function noiseplugin_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to noiseplugin (see VARARGIN)

% Choose default command line output for noiseplugin
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes noiseplugin wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = noiseplugin_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
hand = guidata(handles.hh);
if isempty(hand)  
    msgbox('Can not find main box handles', 'Error');
    return;
end
src = hand.src;

realdata = real(src.y);
realdata = realdata - mean(realdata);
n = size(src.y, 1);

noise = realdata(floor(n/2):end);
signal = datasmooth(realdata,3,'savgol');

fprintf('noiseplugin: std=%f\n', std(noise));
fprintf('noiseplugin: max=%f\n', max(signal));
fprintf('noiseplugin: signal/std/2=%f\n', max(signal)/std(noise)/2);



