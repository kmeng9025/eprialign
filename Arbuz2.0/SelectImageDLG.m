function varargout = SelectImageDLG(varargin)
% SELECTIMAGEDLG M-file for SelectImageDLG.fig
%      SELECTIMAGEDLG, by itself, creates a new SELECTIMAGEDLG or raises the existing
%      singleton*.
%
%      H = SELECTIMAGEDLG returns the handle to a new SELECTIMAGEDLG or the handle to
%      the existing singleton*.
%
%      SELECTIMAGEDLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SELECTIMAGEDLG.M with the given input arguments.
%
%      SELECTIMAGEDLG('Property','Value',...) creates a new SELECTIMAGEDLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before SelectImageDLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to SelectImageDLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help SelectImageDLG

% Last Modified by GUIDE v2.5 04-Feb-2011 14:01:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SelectImageDLG_OpeningFcn, ...
                   'gui_OutputFcn',  @SelectImageDLG_OutputFcn, ...
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


% --- Executes just before SelectImageDLG is made visible.
function SelectImageDLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to SelectImageDLG (see VARARGIN)

% Choose default command line output for SelectImageDLG
handles.output = hObject;
handles.hh = varargin{1};

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes SelectImageDLG wait for user response (see UIRESUME)
uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = SelectImageDLG_OutputFcn(hObject, eventdata, handles) 
varargout{1} = [];
delete(hObject);

% --------------------------------------------------------------------
function figure1_CloseRequestFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)
uiresume(gcbf);

% --------------------------------------------------------------------
function pbSelectOn_Callback(hObject, eventdata, handles)
sel = SelectImages(handles);

arbuz_SetImage(handles.hh, sel, 'Selected', 1);
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbSelectOff_Callback(hObject, eventdata, handles)
sel = SelectImages(handles);

arbuz_SetImage(handles.hh, sel, 'Selected', 0);
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbShowOn_Callback(hObject, eventdata, handles)
sel = SelectImages(handles);

arbuz_SetImage(handles.hh, sel, 'Visible', 1);
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbShowOff_Callback(hObject, eventdata, handles)
sel = SelectImages(handles);

arbuz_SetImage(handles.hh, sel, 'Visible', 0);
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbColor_Callback(hObject, eventdata, handles)
sel = SelectImages(handles);
sel = arbuz_FindImage(handles.hh, sel, '', 0, {'Color'});
if isempty(sel), return; end

color = SetColorDLG(sel{1}.Color);
arbuz_SetImage(handles.hh, sel, 'Color', color);

% Update object list index
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----- S E R V I C E     F U N C T I O N S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function sel = SelectImages(handles)

switch get(handles.pmImageHierarchy, 'Value')
  case 2, % 'master'
    sel = arbuz_FindImage(handles.hh, 'master', '', '', {});
  case 3, % 'proxy'
    sel = arbuz_FindImage(handles.hh, 'proxy', '', '', {});
  case 4, % 'selected'
    sel = arbuz_FindImage(handles.hh, 'all', 'Selected', '', {});
  case 5, % 'visible'
    sel = arbuz_FindImage(handles.hh, 'all', 'Visible', '', {});
  case 6, % 'highlighted'
    sel = arbuz_FindImage(handles.hh, 'all', 'Highlighted', '', {});
  otherwise
    sel = arbuz_FindImage(handles.hh, 'all', '', '', {});
end

if get(handles.cbSelectByType, 'Value')
  switch get(handles.pmCriterionImageType, 'Value')
    case 1 % 2D
      sel = arbuz_FindImage(handles.hh, sel, 'ImageType', '2D', {});
    case 2 % 3DEPRI
      sel = arbuz_FindImage(handles.hh, sel, 'ImageType', '3DEPRI', {});
    case 3 % CONTOUR
      sel = arbuz_FindImage(handles.hh, sel, 'ImageType', 'CONTOUR', {});
    case 4 % 3DEPRI
      sel = arbuz_FindImage(handles.hh, sel, 'ImageType', 'XYZ', {});
    case 5 % 3DSURFACE
      sel = arbuz_FindImage(handles.hh, sel, 'ImageType', '3DSURFACE', {});
  end
end

if get(handles.cbSelectByName, 'Value')
  str = get(handles.eCriterionName, 'String');
  sel = arbuz_FindImage(handles.hh, sel, 'Name', str, {});
end
