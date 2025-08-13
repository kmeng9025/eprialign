function varargout = ProcessValueDialogDLG(varargin)
% PROCESSVALUEDIALOGDLG M-file for ProcessValueDialogDLG.fig
%      PROCESSVALUEDIALOGDLG, by itself, creates a new PROCESSVALUEDIALOGDLG or raises the existing
%      singleton*.
%
%      H = PROCESSVALUEDIALOGDLG returns the handle to a new PROCESSVALUEDIALOGDLG or the handle to
%      the existing singleton*.
%
%      PROCESSVALUEDIALOGDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROCESSVALUEDIALOGDLG.M with the given input arguments.
%
%      PROCESSVALUEDIALOGDLG('Property','Value',...) creates a new PROCESSVALUEDIALOGDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ProcessValueDialogDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ProcessValueDialogDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ProcessValueDialogDLG

% Last Modified by GUIDE v2.5 23-Aug-2017 15:12:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ProcessValueDialogDLG_OpeningFcn, ...
                   'gui_OutputFcn',  @ProcessValueDialogDLG_OutputFcn, ...
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

% --------------------------------------------------------------------
function ProcessValueDialogDLG_OpeningFcn(hObject, eventdata, handles, varargin)

if isempty(varargin)
  closereq;
  return;
end

handles.pars = varargin{1};
handles.fields = varargin{2};
par  = varargin{3};
handles.ini =  varargin{4};

par.ControlTMP = par.Control;

set(handles.uipanelvalue, 'Title', sprintf('Value (%s.%s)',par.Group, par.Field))
set([handles.eValue, handles.pmValue, handles.pbValueFSelect], 'Visible', 'off');
switch par.Type
  case {'D', 'S', 'F'}
    par.Control = handles.eValue;
    set(handles.eValue, 'Visible', 'on', 'UserData', par)
    ProcessFormController(handles.pars, handles.fields, par,'ini2gui');
    if par.Type == 'F'
      set(handles.pbValueFSelect, 'Visible', 'on', 'UserData', par)
    end
  case {'IDX','IDX1+','IDXS'}
    par.Control = handles.pmValue;
    set(handles.pmValue, 'String', par.Show, 'Value', 1, 'UserData', par, 'Visible', 'on')    
    ProcessFormController(handles.pars, handles.fields, par,'ini2gui');
end

% Update handles structure
guidata(hObject, handles);
uiwait(handles.figure1)


% --------------------------------------------------------------------
function varargout = ProcessValueDialogDLG_OutputFcn(hObject, eventdata, handles) 
if isfield(handles, 'OptionsOk')
  handles = guidata(hObject);

  varargout{1} = handles.pars;
  varargout{2} = handles.fields;
  varargout{3} = handles.OptionsOk;
  delete(hObject);
else
  varargout{1} = {};
  varargout{2} = {};
  varargout{3} = 0;
end

% --------------------------------------------------------------------
function pbOk_Callback(hObject, eventdata, handles)
handles.OptionsOk = 1;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)
handles.OptionsOk = 0;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pmValue_Callback(hObject, eventdata, handles)
par = get(handles.pmValue, 'UserData');
[handles.pars, handles.fields] = ...
  ProcessFormController(handles.pars, handles.fields, par,'gui2ini');
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function eValue_Callback(hObject, eventdata, handles)
par = get(handles.eValue, 'UserData');

[handles.pars, handles.fields] = ...
  ProcessFormController(handles.pars, handles.fields, par,'gui2ini');
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbValueFSelect_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/'); 

[FileName,PathName] = uigetfile({'*.mat', 'Matlab files (*.mat)'; '*.*', 'All files (*.*)'},'Open file', old_path, ...
  'MultiSelect', 'off');
if PathName ~= 0
  set(handles.eValue, 'String', fullfile(PathName, FileName));
  eValue_Callback(handles.eValue, eventdata, handles)
end
