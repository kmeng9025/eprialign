function varargout = EditNotesPLG(varargin)
% EDITNOTESPLG M-file for EditNotesPLG.fig
%      EDITNOTESPLG, by itself, creates a new EDITNOTESPLG or
%      raises the existing
%      singleton*.
%
%      H = EDITNOTESPLG returns the handle to a new EDITNOTESPLG or the handle to
%      the existing singleton*.
%
%      EDITNOTESPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in EDITNOTESPLG.M with the given input arguments.
%
%      EDITNOTESPLG('Property','Value',...) creates a new EDITNOTESPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before EditNotesPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to EditNotesPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help EditNotesPLG

% Last Modified by GUIDE v2.5 29-Mar-2016 13:00:00

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @EditNotesPLG_OpeningFcn, ...
  'gui_OutputFcn',  @EditNotesPLG_OutputFcn, ...
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
function EditNotesPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for Image2ContourPLG
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};

% Update handles structure
guidata(hObject, handles);

comments =arbuz_get(handles.hh, 'comments');
set(handles.eNotes, 'String', comments);

% --------------------------------------------------------------------
function varargout = EditNotesPLG_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;


% --- Executes on button press in pbOK.
function pbOK_Callback(hObject, eventdata, handles)
comments = get(handles.eNotes, 'String');
arbuz_set(handles.hh, 'comments', comments);


% --- Executes on button press in pbCancel.
function pbCancel_Callback(hObject, eventdata, handles)
comments = get(handles.eNotes, 'String');
arbuz_set(handles.hh, 'comments', comments);
delete(handles.figure1);


% --- Executes on button press in pbRefresh.
function pbRefresh_Callback(hObject, eventdata, handles)
comments =arbuz_get(handles.hh, 'comments');
set(handles.eNotes, 'String', comments);
