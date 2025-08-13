function varargout = pvExpressionEditor(varargin)
% PVEXPRESSIONEDITOR MATLAB code for pvExpressionEditor.fig
%      PVEXPRESSIONEDITOR, by itself, creates a new PVEXPRESSIONEDITOR or raises the existing
%      singleton*.
%
%      H = PVEXPRESSIONEDITOR returns the handle to a new PVEXPRESSIONEDITOR or the handle to
%      the existing singleton*.
%
%      PVEXPRESSIONEDITOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PVEXPRESSIONEDITOR.M with the given input arguments.
%
%      PVEXPRESSIONEDITOR('Property','Value',...) creates a new PVEXPRESSIONEDITOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pvExpressionEditor_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pvExpressionEditor_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pvExpressionEditor

% Last Modified by GUIDE v2.5 04-Oct-2016 11:09:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pvExpressionEditor_OpeningFcn, ...
                   'gui_OutputFcn',  @pvExpressionEditor_OutputFcn, ...
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


% --- Executes just before pvExpressionEditor is made visible.
function pvExpressionEditor_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pvExpressionEditor (see VARARGIN)

% Choose default command line output for pvExpressionEditor
handles.output = hObject;
handles.isOK = false;

if nargin > 4
  set(handles.pmFields, 'string', varargin{2})
end
% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pvExpressionEditor wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = pvExpressionEditor_OutputFcn(hObject, eventdata, handles)
varargout{1} = '';
try
  if handles.isOK
    varargout{1} = get(handles.eExpression, 'string');
  end
catch
end
delete(hObject);


% --- Executes on button press in pbAddField.
function pbAddField_Callback(hObject, eventdata, handles)
mycursor = 1;

val = get(handles.pmFields, 'value');
str = get(handles.pmFields, 'string');

mystring = get(handles.eExpression, 'string');
outstring = [mystring(1:mycursor-1), str{val}, mystring(mycursor:end)];
set(handles.eExpression, 'string', outstring);

% --- Executes on button press in pbApply.
function pbApply_Callback(hObject, eventdata, handles)
handles.isOK = true;
guidata(hObject, handles);
uiresume(gcbf);


% --- Executes on button press in pbCancel.
function pbCancel_Callback(hObject, eventdata, handles)
handles.isOK = false;
guidata(hObject, handles);
uiresume(gcbf);
