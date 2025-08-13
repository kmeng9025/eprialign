function varargout = LoadImageListGUI(varargin)
% LOADIMAGELISTGUI MATLAB code for LoadImageListGUI.fig
%      LOADIMAGELISTGUI, by itself, creates a new LOADIMAGELISTGUI or raises the existing
%      singleton*.
%
%      H = LOADIMAGELISTGUI returns the handle to a new LOADIMAGELISTGUI or the handle to
%      the existing singleton*.
%
%      LOADIMAGELISTGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LOADIMAGELISTGUI.M with the given input arguments.
%
%      LOADIMAGELISTGUI('Property','Value',...) creates a new LOADIMAGELISTGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LoadImageListGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LoadImageListGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LoadImageListGUI

% Last Modified by GUIDE v2.5 24-Mar-2011 14:37:39

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LoadImageListGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @LoadImageListGUI_OutputFcn, ...
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
function LoadImageListGUI_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for LoadImageListGUI
handles.output = hObject;
handles.ini.Directories = [];
handles.Timeline = {};

if nargin > 0
  handles.Timeline = varargin{1};
end
if nargin > 1
  handles.ini.Directories.SourcePath = varargin{2};
end

% Update handles structure
guidata(hObject, handles);
PrepareList(handles);

uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = LoadImageListGUI_OutputFcn(hObject, eventdata, handles) 
if isfield(handles, 'FileListOk')
 varargout{1} = handles.Timeline;
 varargout{2} = handles.FileListOk;
 delete(hObject);
else
 varargout{1} = {};
 varargout{2} = {};
end

% --------------------------------------------------------------------
function pbDn_Callback(hObject, eventdata, handles)
nItem = get(handles.lbFiles, 'Value');
nAll = length(handles.Timeline);

if nItem < nAll
  timeline_idx = 1:nAll;
  handles.Timeline = handles.Timeline(timeline_idx([1:nItem-1,nItem+1,nItem,nItem+2:nAll]));
  guidata(hObject, handles);
  PrepareList(handles);
  set(handles.lbFiles, 'Value', nItem+1);
end

% --------------------------------------------------------------------
function pbUp_Callback(hObject, eventdata, handles)
nItem = get(handles.lbFiles, 'Value');
nAll = length(handles.Timeline);

if nItem > 1
  timeline_idx = 1:nAll;
  handles.Timeline = handles.Timeline(timeline_idx([1:nItem-2,nItem,nItem-1,nItem+1:nAll]));
  guidata(hObject, handles);
  PrepareList(handles);
  set(handles.lbFiles, 'Value', nItem-1);
end

% --------------------------------------------------------------------
function lbFiles_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)
handles.FileListOk = 1;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)
handles.FileListOk = 0;
guidata(hObject, handles);
uiresume

% --------------------------------------------------------------------
function pbAdd_Callback(hObject, eventdata, handles)
old_path     = safeget(handles.ini.Directories, 'SourcePath', '');

[FileName,PathName] = uigetfile({'*.mat', 'Matlab files (*.mat)'},'Load file', old_path, ...
  'MultiSelect', 'on');
if PathName ~= 0
  handles.ini.Directories.SourcePath = PathName;
  guidata(handles.figure1, handles);
  if ~iscell(FileName), FileName = {FileName}; end
  for ii=1:length(FileName)
    handles.Timeline{end+1}.FileName = fullfile(PathName, FileName{ii});
  end
  guidata(hObject, handles);
  PrepareList(handles);
end

% --------------------------------------------------------------------
function pbRemove_Callback(hObject, eventdata, handles)
nItem = get(handles.lbFiles, 'Value');
nAll = length(handles.Timeline);

if nItem <= nAll
  timeline_idx = 1:nAll;
  handles.Timeline = handles.Timeline(timeline_idx([1:nItem-1,nItem+1:nAll]));
  guidata(hObject, handles);
  PrepareList(handles);
end

% --------------------------------------------------------------------
function str = PrepareList(handles)
str = {};
for ii=1:length(handles.Timeline)
  str{end+1} = handles.Timeline{ii}.FileName;
end
if isempty(str), str = {'None'}; end
lastval = get(handles.lbFiles, 'Value');
set(handles.lbFiles, 'String', str, 'Value', max([1, min([lastval, length(handles.lbFiles)])]));
