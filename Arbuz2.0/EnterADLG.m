function varargout = EnterADLG(varargin)
% ENTERADLG MATLAB code for EnterADLG.fig
%      ENTERADLG, by itself, creates a new ENTERADLG or raises the existing
%      singleton*.
%
%      H = ENTERADLG returns the handle to a new ENTERADLG or the handle to
%      the existing singleton*.
%
%      ENTERADLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in ENTERADLG.M with the given input arguments.
%
%      ENTERADLG('Property','Value',...) creates a new ENTERADLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EnterADLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EnterADLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EnterADLG

% Last Modified by GUIDE v2.5 09-Jun-2021 14:27:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EnterADLG_OpeningFcn, ...
                   'gui_OutputFcn',  @EnterADLG_OutputFcn, ...
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


% --- Executes just before EnterADLG is made visible.
function EnterADLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to EnterADLG (see VARARGIN)

% Choose default command line output for EnterADLG
handles.output = hObject;

handles.hEdit = zeros(3);
handles.hText = zeros(3);
for ii=1:4
  for jj=1:4
    handles.hEdit(jj,ii) = uicontrol('Style','edit', 'units', 'normalized', 'string', 0.0);
    handles.hText(jj,ii) = uicontrol('Style','text', 'units', 'normalized',...
      'HorizontalAlignment', 'left', 'string', sprintf('(%i,%i)', jj, ii));
  end
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes EnterADLG wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = EnterADLG_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
delete(hObject);

% --- Executes on button press in pbSET.
function pbSET_Callback(hObject, eventdata, handles)
handles.output = [];
for ii=1:4
  for jj=1:4
    handles.output(jj,ii) = eval(get(handles.hEdit(jj,ii), 'string'));
  end
end
guidata(hObject, handles);
uiresume(gcbf)


% --- Executes on button press in pbCANCEL.
function pbCANCEL_Callback(hObject, eventdata, handles)
handles.output = [];
guidata(hObject, handles);
uiresume(gcbf);


% --- Executes when figure1 is resized.
function figure1_SizeChangedFcn(hObject, eventdata, handles)
for ii=1:4
  for jj=1:4
    px = 0.02+(ii-1)*0.25;
    py = 1-jj*0.17;
    set(handles.hText(jj,ii), 'Position', [px, py, 0.06, 0.1]);
    set(handles.hEdit(jj,ii), 'Position', [px+0.06, py+0.025, 0.16, 0.1]);
  end
end

% --- Executes on button press in pbLoadWorkspaceA.
function pbLoadWorkspaceA_Callback(hObject, eventdata, handles)
A = evalin('base', 'A');
for ii=1:4
  for jj=1:4
    set(handles.hEdit(jj,ii), 'string', A(jj,ii));
  end
end
