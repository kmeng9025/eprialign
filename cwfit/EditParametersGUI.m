function varargout = EditParametersGUI(varargin)
% EDITPARAMETERSGUI M-file for EditParametersGUI.fig
%      EDITPARAMETERSGUI, by itself, creates a new EDITPARAMETERSGUI or raises the existing
%      singleton*.
%
%      H = EDITPARAMETERSGUI returns the handle to a new EDITPARAMETERSGUI or the handle to
%      the existing singleton*.
%
%      EDITPARAMETERSGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EDITPARAMETERSGUI.M with the given input arguments.
%
%      EDITPARAMETERSGUI('Property','Value',...) creates a new EDITPARAMETERSGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EditParametersGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EditParametersGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EditParametersGUI

% Last Modified by GUIDE v2.5 19-Nov-2008 10:32:31

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @EditParametersGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @EditParametersGUI_OutputFcn, ...
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
function EditParametersGUI_OpeningFcn(hObject, eventdata, handles, varargin)

handles.output = hObject;

if ~isempty(varargin)
  handles.Parameters = varargin{1};
  d = handles.Parameters{1};
  MaxCol = size(d, 2);
else
  handles.Parameters = {};
  MaxCol = 1;
end

if nargin > 4, 
  flds = {'ColumnName', 'ColumnFormat', 'ColumnEditable',...
    'ColumnWidth','SetRow4All','Set','Set4All'};
  for ii=1:length(flds)
    if isfield(varargin{2}, flds{ii})
      handles = setfield(handles, flds{ii}, getfield(varargin{2}, flds{ii}));
    end
  end
end
if ~isfield(handles, 'Set'), handles.Set= 1; end
if ~isfield(handles, 'Set4All'), handles.Set4All=zeros(MaxCol,1); end
if ~isfield(handles, 'SetRow4All'), handles.SetRow4All=zeros(MaxCol,1); end
if ~isfield(handles, 'ColumnWidth'), handles.ColumnWidth={50, 50}; end
  
set(handles.uitable1, 'ColumnName', handles.ColumnName);
set(handles.uitable1, 'ColumnFormat', handles.ColumnFormat);
set(handles.uitable1, 'ColumnEditable', handles.ColumnEditable);

if length(handles.Parameters)==1
  set(handles.pbNext, 'Visible', 'off');
  set(handles.pbPrevious, 'Visible', 'off');
end

set(handles.uitable1, 'Units', 'pixels');
ChangeSet(handles, -1, handles.Set)

% Update handles structure
guidata(hObject, handles);
uiwait(handles.figure1)

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
p = get(handles.figure1, 'Position');
pp = [5, 50, p(3)-10, p(4)-55];
set(handles.uitable1, 'Position', pp);

if isfield(handles, 'Parameters')
  width = 0;
  for ii=2:length(handles.ColumnWidth), width = width + handles.ColumnWidth{ii}; end
  handles.ColumnWidth{1}=pp(3)-25-width;
  set(handles.uitable1, 'ColumnWidth', handles.ColumnWidth);
end

% --------------------------------------------------------------------
function varargout = EditParametersGUI_OutputFcn(hObject, eventdata, handles) 
if isfield(handles, 'FileListOk')
 varargout{1} = handles.Parameters;
 varargout{2} = handles.FileListOk;
 delete(hObject);
else
 varargout{1} = {};
 varargout{2} = 0;
end

% --------------------------------------------------------------------
function pbOk_Callback(hObject, eventdata, handles)
handles.FileListOk = 1;
ChangeSet(handles, handles.Set, -1);
uiresume

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)
handles.FileListOk = 0;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pbPrevious_Callback(hObject, eventdata, handles)
OldSet = handles.Set;
MaxSet = length(handles.Parameters);
if handles.Set == 1, handles.Set = MaxSet; else handles.Set = handles.Set - 1; end
ChangeSet(handles, OldSet, handles.Set);

% --------------------------------------------------------------------
function pbNext_Callback(hObject, eventdata, handles)
OldSet = handles.Set;
MaxSet = length(handles.Parameters);
if handles.Set == MaxSet, handles.Set = 1; else handles.Set = handles.Set + 1; end
ChangeSet(handles, OldSet, handles.Set);

% --------------------------------------------------------------------
function eSet_Callback(hObject, eventdata, handles)
OldSet = handles.Set;
MaxSet = length(handles.Parameters);
handles.Set = fix(str2num(get(handles.eSet, 'String')));
if handles.Set < 1, handles.Set = 1; end
if handles.Set > MaxSet, handles.Set = MaxSet; end
ChangeSet(handles, OldSet, handles.Set);

% --------------------------------------------------------------------
function uitable1_CellEditCallback(hObject, eventdata, handles)

switch CopyOrNot(handles, eventdata.Indices)
  case 1 % copy row
    d1 = get(handles.uitable1, 'Data');
    for ii=1:length(handles.Parameters)
      d = handles.Parameters{ii};
      for jj=1:size(d1,2)
        d{eventdata.Indices(1,1), jj} = d1{eventdata.Indices(1,1), jj};
      end
      handles.Parameters{ii} = d;
    end
    guidata(hObject, handles);
  case 2 %copy value
    for ii=1:length(handles.Parameters)
      d = handles.Parameters{ii};
      d{eventdata.Indices(1,1), eventdata.Indices(1,2)} = eventdata.NewData;
      handles.Parameters{ii} = d;
    end
    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function answer = CopyOrNot(handles, idx)
answer = 0; % do not copy
d = get(handles.uitable1, 'Data');
c_idx = find(handles.SetRow4All);
if isempty(c_idx), return; end
if c_idx(1) == idx(2) && d{idx(1), idx(2)}, answer = 1; return; end % copy row
if d{idx(1), c_idx(1)} || handles.Set4All(idx(1,2)) > 0, answer = 2; end % copy value
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function ChangeSet(handles, set1, set2)

if set1 > 0
  handles.Parameters{set1} = get(handles.uitable1, 'Data');
  guidata(handles.figure1, handles);
end

if set2 > 0
  set(handles.uitable1, 'Data', handles.Parameters{set2})
  set(handles.eSet, 'String', num2str(set2))
  
  % resize
  pp = get(handles.figure1, 'Position');
  new_high = length(handles.Parameters{set2}) * 20 + 80;
  if new_high > pp(4)
    p = get(0, 'MonitorPositions');
    pp(4) = new_high; 
    if pp(2)+pp(4) > min(p(:,4)), pp(2) = min(p(:,4)) - pp(4) - 30; end
    set(handles.figure1, 'Position', pp);
  end
end
