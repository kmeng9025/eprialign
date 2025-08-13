function varargout = SetCoordinateSystem(varargin)
% SETCOORDINATESYSTEM MATLAB code for SetCoordinateSystem.fig
%      SETCOORDINATESYSTEM, by itself, creates a new SETCOORDINATESYSTEM or raises the existing
%      singleton*.
%
%      H = SETCOORDINATESYSTEM returns the handle to a new SETCOORDINATESYSTEM or the handle to
%      the existing singleton*.
%
%      SETCOORDINATESYSTEM('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SETCOORDINATESYSTEM.M with the given input arguments.
%
%      SETCOORDINATESYSTEM('Property','Value',...) creates a new SETCOORDINATESYSTEM or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SetCoordinateSystem_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SetCoordinateSystem_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SetCoordinateSystem

% Last Modified by GUIDE v2.5 30-Jan-2015 11:37:20

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SetCoordinateSystem_OpeningFcn, ...
                   'gui_OutputFcn',  @SetCoordinateSystem_OutputFcn, ...
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


% --- Executes just before SetCoordinateSystem is made visible.
function SetCoordinateSystem_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SetCoordinateSystem (see VARARGIN)

% Choose default command line output for SetCoordinateSystem
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SetCoordinateSystem wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --------------------------------------------------------------------
function varargout = SetCoordinateSystem_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.results;
delete(handles.figure1);

% --------------------------------------------------------------------
function pmReference_Callback(hObject, eventdata, handles)

imageXYZ = [0,0,0];
switch get(handles.pmReference, 'Value')
  case 1
    xyz = imageXYZ;
  case 2
    xyz = [-12, 12, 12];
end

set(handles.eRefX, 'string',  num2str(xyz(1)));
set(handles.eRefY, 'string',  num2str(xyz(2)));
set(handles.eRefZ, 'string',  num2str(xyz(3)));

% --------------------------------------------------------------------
function pbOK_Callback(hObject, eventdata, handles)

Rxyz(1) = str2double(get(handles.eRefX, 'string'));
Rxyz(2) = str2double(get(handles.eRefY, 'string'));
Rxyz(3) = str2double(get(handles.eRefZ, 'string'));

Nxyz(1) = str2double(get(handles.eRefX, 'string'));
Nxyz(2) = str2double(get(handles.eRefY, 'string'));
Nxyz(3) = str2double(get(handles.eRefZ, 'string'));

handles.results.Rxyz = Rxyz;
handles.results.Nxyz = Nxyz;
guidata(hObject, handles);

uiresume(handles.figure1);

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)

handles.results.xyz = [];
guidata(hObject, handles);

uiresume(handles.figure1);

% --------------------------------------------------------------------
function pmNewCoord_Callback(hObject, eventdata, handles)
