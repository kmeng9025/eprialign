function varargout = EditSequenceDLG(varargin)
% EDITSEQUENCEDLG M-file for EditSequenceDLG.fig
%      EDITSEQUENCEDLG, by itself, creates a new EDITSEQUENCEDLG or raises the existing
%      singleton*.
%
%      H = EDITSEQUENCEDLG returns the handle to a new EDITSEQUENCEDLG or the handle to
%      the existing singleton*.
%
%      EDITSEQUENCEDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EDITSEQUENCEDLG.M with the given input arguments.
%
%      EDITSEQUENCEDLG('Property','Value',...) creates a new EDITSEQUENCEDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EditSequenceDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EditSequenceDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EditSequenceDLG

% Last Modified by GUIDE v2.5 04-Feb-2011 13:36:19

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @EditSequenceDLG_OpeningFcn, ...
  'gui_OutputFcn',  @EditSequenceDLG_OutputFcn, ...
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
function EditSequenceDLG_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
% Choose default command line output for EditSequenceDLG
handles.output = hObject;
handles.hh = varargin{1};
handles.ActiveSequence = varargin{2};

Sequences = arbuz_get(handles.hh, 'Sequences');
handles.Sequence = Sequences{handles.ActiveSequence};
handles.Transformations = arbuz_get(handles.hh, 'Transformations');
SetTransformationList(handles);

set(handles.figure1, 'Name', ['EditSequence [',handles.Sequence.Name,']']);
handles.ActiveTransformation = safeget(handles.Sequence, 'ActiveTransformation', 1);
handles.WatchTransformation  = safeget(handles.Sequence, 'WatchTransformation', 1);
handles.isOk = 0;

RedrawSequence(handles);

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes EditSequenceDLG wait for user response (see UIRESUME)
uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = EditSequenceDLG_OutputFcn(hObject, eventdata, handles)
seq = handles.Sequence;
seq.ActiveTransformation = handles.ActiveTransformation;
seq.WatchTransformation = handles.WatchTransformation;

varargout{1} = seq;
varargout{2} = handles.isOk;

delete(hObject);

% --------------------------------------------------------------------
function pbUp_Callback(hObject, eventdata, handles)
posS = get(handles.lbSequence, 'Value');
if posS < 2, return; end
idx = 1:length(handles.Sequence.Sequence);
idx(posS-1:posS) = idx(posS:-1:posS-1);
handles.Sequence.Sequence = handles.Sequence.Sequence(idx);
guidata(handles.figure1, handles);
set(handles.lbSequence, 'Value',posS-1);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbDn_Callback(hObject, eventdata, handles)
posS = get(handles.lbSequence, 'Value');
if posS < 0 || posS > length(handles.Sequence.Sequence) - 1, return; end
idx = 1:length(handles.Sequence.Sequence);
idx(posS:posS+1) = idx(posS+1:-1:posS);
handles.Sequence.Sequence = handles.Sequence.Sequence(idx);
guidata(handles.figure1, handles);
set(handles.lbSequence, 'Value',posS+1);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbUnassignTransformation_Callback(hObject, eventdata, handles)

posS = get(handles.lbSequence, 'Value');
if posS == 0, return; end

handles.Sequence.Sequence{posS}.Name = '';

guidata(handles.figure1, handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbAssignTransformation_Callback(hObject, eventdata, handles) %#ok<*DEFNU>

posT = get(handles.lbTransformations, 'Value');
strT = get(handles.lbTransformations, 'String');

posS = get(handles.lbSequence, 'Value');
posS = max(posS, 1);

handles.Sequence.Sequence{posS}.Name = strT{posT};

guidata(handles.figure1, handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbVisual_Callback(hObject, eventdata, handles)

handles.WatchTransformation = get(handles.lbSequence, 'Value');
guidata(handles.figure1, handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbActivate_Callback(hObject, eventdata, handles)

handles.ActiveTransformation = get(handles.lbSequence, 'Value');
guidata(handles.figure1, handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbOk_Callback(hObject, eventdata, handles)
handles.isOk = 1;
guidata(handles.figure1, handles);

uiresume(gcbf);

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles) %#ok<*INUSD>
uiresume(gcbf);

% --------------------------------------------------------------------
function pbIns_Callback(hObject, eventdata, handles)

n = length(handles.Sequence.Sequence);
answer = inputdlg({['Description for stage ', num2str(n+1)]},'Stage parameters',1, {'Enter description'});
if ~isempty(answer)
  handles.Sequence.Sequence{end+1} = struct('Name','','Description',answer{1});
  if length(handles.Sequence.Sequence) == 1
    handles.ActiveTransformation = 1;
    handles.WatchTransformation = 1;
  end
  guidata(handles.figure1, handles);
  RedrawSequence(handles);
end

% --------------------------------------------------------------------
function pbDel_Callback(hObject, eventdata, handles)

posS = get(handles.lbSequence, 'Value');
if posS == 0, return; end
handles.Sequence.Sequence = handles.Sequence.Sequence([1:posS-1,posS+1:end]);
guidata(handles.figure1, handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
function pbSet_Callback(hObject, eventdata, handles)

posS = get(handles.lbSequence, 'Value');
if posS == 0, return; end

new_dsc = get(handles.eDescription, 'String');

handles.Sequence.Sequence{posS}.Description = new_dsc;
guidata(handles.figure1, handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
function lbSequence_Callback(hObject, eventdata, handles)
pos = get(handles.lbSequence, 'Value');
if ~isempty(handles.Sequence.Sequence)
  set(handles.eDescription, 'String', safeget(handles.Sequence.Sequence{pos},'Description', ''));
end

% --------------------------------------------------------------------
function pbInsTransformation_Callback(hObject, eventdata, handles)
answer = inputdlg('Type the name of new transformation', 'Input', 1, {'transformation'});

if ~isempty(answer)
  arbuz_AddTransformation(handles.hh, answer{1});
  
  handles.Transformations = arbuz_get(handles.hh, 'Transformations');
  guidata(handles.figure1, handles);
  SetTransformationList(handles);
end

% --------------------------------------------------------------------
function pbDeleteTransformation_Callback(hObject, eventdata, handles)
posT = get(handles.lbTransformations, 'Value');
strT = get(handles.lbTransformations, 'String');
if posT < 1, return; end

arbuz_DeleteTransformation(handles.hh, strT{posT});
% reload everything
handles.Transformations = arbuz_get(handles.hh, 'Transformations');
Sequences = arbuz_get(handles.hh, 'Sequences');
handles.Sequence = Sequences{handles.ActiveSequence};

guidata(handles.figure1, handles);
SetTransformationList(handles);
RedrawSequence(handles);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----- S E R V I C E     F U N C T I O N S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function RedrawSequence(handles)

str = {};
for ii=1:length(handles.Sequence.Sequence)
  item_str = ['Stage ',num2str(ii),' (',handles.Sequence.Sequence{ii}.Name,')'];
  
  if ii == handles.WatchTransformation
    item_str = ['<b><font color=red>',item_str,'</font></b>'];
  else
    item_str = item_str;
  end
  
  if ii == handles.ActiveTransformation
    str{end+1} = ['<html>&nbsp;+',item_str,'</html>'];
  else
    str{end+1} = ['<html>&nbsp;&nbsp;',item_str,'</html>'];
  end
end

pos = max(1, get(handles.lbSequence, 'Value'));
pos = min(pos, length(str));
set(handles.lbSequence, 'String', str, 'Value',pos);
if ~isempty(handles.Sequence.Sequence)
  set(handles.eDescription, 'String', safeget(handles.Sequence.Sequence{pos},'Description', ''));
end

% --------------------------------------------------------------------
function SetTransformationList(handles)
posT = get(handles.lbTransformations, 'Value');
nT = length(handles.Transformations);
str = cell(1,nT);
for ii=1:nT
  str{ii} = handles.Transformations{ii}.Name;
end
set(handles.lbTransformations, 'String', str, 'Value', max(min(posT,nT),1));


