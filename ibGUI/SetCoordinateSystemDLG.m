function varargout = SetCoordinateSystemDLG(varargin)
% SETCOORDINATESYSTEMDLG MATLAB code for SetCoordinateSystemDLG.fig
%      SETCOORDINATESYSTEMDLG, by itself, creates a new SETCOORDINATESYSTEMDLG or raises the existing
%      singleton*.
%
%      H = SETCOORDINATESYSTEMDLG returns the handle to a new SETCOORDINATESYSTEMDLG or the handle to
%      the existing singleton*.
%
%      SETCOORDINATESYSTEMDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SETCOORDINATESYSTEMDLG.M with the given input arguments.
%
%      SETCOORDINATESYSTEMDLG('Property','Value',...) creates a new SETCOORDINATESYSTEMDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SetCoordinateSystemDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SetCoordinateSystemDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SetCoordinateSystemDLG

% Last Modified by GUIDE v2.5 30-Jan-2015 13:03:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SetCoordinateSystemDLG_OpeningFcn, ...
                   'gui_OutputFcn',  @SetCoordinateSystemDLG_OutputFcn, ...
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


% --- Executes just before SetCoordinateSystemDLG is made visible.
function SetCoordinateSystemDLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SetCoordinateSystemDLG (see VARARGIN)

% Choose default command line output for SetCoordinateSystemDLG
handles.output = hObject;

handles.imageXYZ = varargin{1}.ImageSize;

set(handles.eRefX, 'string',  num2str(varargin{1}.REFxyz(1)));
set(handles.eRefY, 'string',  num2str(varargin{1}.REFxyz(2)));
set(handles.eRefZ, 'string',  num2str(varargin{1}.REFxyz(3)));

set(handles.eNewCoordX, 'string',  num2str(varargin{1}.NCxyz(1)));
set(handles.eNewCoordY, 'string',  num2str(varargin{1}.NCxyz(2)));
set(handles.eNewCoordZ, 'string',  num2str(varargin{1}.NCxyz(3)));

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SetCoordinateSystemDLG wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --------------------------------------------------------------------
function varargout = SetCoordinateSystemDLG_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.results;
delete(handles.figure1);

% --------------------------------------------------------------------
function pmReference_Callback(hObject, eventdata, handles)

switch get(handles.pmReference, 'Value')
  case 1 % image center
    xyz = [0,0,0];
  case 2 % 19 mm resonator in IM
    xyz = [1.9/2, -1.5/2, 0];
  case 3 % 25 mm resonator in IM
    xyz = [2.5/2, -2.5/2, 0];
end

set(handles.eRefX, 'string',  num2str(xyz(1)));
set(handles.eRefY, 'string',  num2str(xyz(2)));
set(handles.eRefZ, 'string',  num2str(xyz(3)));

% --------------------------------------------------------------------
function pbOK_Callback(hObject, eventdata, handles)

Rxyz(1) = str2double(get(handles.eRefX, 'string'));
Rxyz(2) = str2double(get(handles.eRefY, 'string'));
Rxyz(3) = str2double(get(handles.eRefZ, 'string'));

Nxyz(1) = str2double(get(handles.eNewCoordX, 'string'));
Nxyz(2) = str2double(get(handles.eNewCoordY, 'string'));
Nxyz(3) = str2double(get(handles.eNewCoordZ, 'string'));

handles.results.REFxyz = Rxyz;
handles.results.NCxyz = Nxyz;
guidata(hObject, handles);

uiresume(handles.figure1);

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)

handles.results.REFxyz = [];
handles.results.NCxyz = [];
guidata(hObject, handles);

uiresume(handles.figure1);

% --------------------------------------------------------------------
function pmNewCoord_Callback(hObject, eventdata, handles)
