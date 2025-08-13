function varargout = RotateImagePLG(varargin)
% ROTATEIMAGEPLG M-file for RotateImagePLG.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2007

% Last Modified by GUIDE v2.5 22-Apr-2023 23:58:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @RotateImagePLG_OpeningFcn, ...
  'gui_OutputFcn',  @RotateImagePLG_OutputFcn, ...
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


% --- Executes just before RotateImagePLG is made visible.
function RotateImagePLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to RotateImagePLG (see VARARGIN)

% javax.swing.UIManager.setLookAndFeel('javax.swing.plaf.metal.MetalLookAndFeel');

% Choose default command line output for RotateImagePLG
handles.output = hObject;
handles.hh = varargin{1};

handles.TranslateMode = 'X';
handles.RotateMode    = 'XY';
handles.ScaleMode     = 'x';
handles.MirrorMode    = '-';

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];
handles.Scale     = [1,1,1];

handles.AAA = {};
handles.LastTransformation = '?';

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = RotateImagePLG_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
if isempty(handles), return; end

xborder = 1.5;

row_space = 2.2;
row1 = 10.1; row2 = row1 - row_space; row3 = row2 - row_space; row4 = row3 - row_space;

xbutton_size  = 6.5;
ybutton_size  = 2.0;
xbutton_space = 1.5;
buttons_rotate = [handles.pbRotateXY, handles.pbRotateXZ, handles.pbRotateYZ];
buttons_translate = [handles.pbTranslateX, handles.pbTranslateY, handles.pbTranslateZ];
buttons_scale = [handles.pbScaleX, handles.pbScaleY, handles.pbScaleZ, handles.pbScaleAll];
buttons_mirror = [handles.pbMirrorNo, handles.pbMirrorX, handles.pbMirrorY, handles.pbMirrorZ];

xscale_shift = 27;
xscale_size = 10;

xslider_shift = xscale_shift + xscale_size + 1;
xslider_size = 21;

xedit_shift = xslider_shift + xslider_size + 1;
xedit_size = 10;

% buttons
for ii=1:length(buttons_rotate)
  set(buttons_rotate(ii), 'Position', [xborder + (ii-1)*(xbutton_size+xbutton_space), row1, xbutton_size, ybutton_size]);
end
for ii=1:length(buttons_translate)
  set(buttons_translate(ii), 'Position', [xborder + (ii-1)*(xbutton_size+xbutton_space), row2, xbutton_size, ybutton_size]);
end
for ii=1:length(buttons_scale)
  set(buttons_scale(ii), 'Position', [xborder + (ii-1)*(xbutton_size*0.8+xbutton_space*0.4), row3, xbutton_size*0.8, ybutton_size]);
end
for ii=1:length(buttons_mirror)
  set(buttons_mirror(ii), 'Position', [xborder + (ii-1)*(xbutton_size+xbutton_space), row4, xbutton_size, ybutton_size]);
end

% scales
set(handles.pmRotateStep, 'Position', [xscale_shift, row1, xscale_size, ybutton_size])
set(handles.pmTranslateStep, 'Position', [xscale_shift, row2, xscale_size, ybutton_size])

% sliders
set(handles.slRotate, 'Position', [xslider_shift, row1, xslider_size, ybutton_size])
set(handles.slTranslate, 'Position', [xslider_shift, row2, xslider_size, ybutton_size])
set(handles.slScale, 'Position', [xslider_shift, row3, xslider_size, ybutton_size])

% sliders
set(handles.eRotate, 'Position', [xedit_shift, row1, xedit_size, ybutton_size])
set(handles.eTranslate, 'Position', [xedit_shift, row2, xedit_size, ybutton_size])
set(handles.eScale, 'Position', [xedit_shift, row3, xedit_size, ybutton_size])


% --------------------------------------------------------------------
function slRotate_Callback(hObject, eventdata, handles)

str  = get(handles.pmRotateStep, 'String');
val  = get(handles.pmRotateStep, 'Value');
step = str2double(str(val,:));

val   = get(hObject,'Value');
pos = GetRotation(handles);

SetRotation(handles, pos + step*sign(val), true);
set(hObject,'Value',0);

% --------------------------------------------------------------------
function slTranslate_Callback(hObject, eventdata, handles)

str  = get(handles.pmTranslateStep, 'String');
val  = get(handles.pmTranslateStep, 'Value');
step = str2double(str{val});

val   = get(hObject,'Value');
pos = GetTranslation(handles);

SetTranslation(handles, pos + step*sign(val), true);
set(hObject,'Value',0);

% --------------------------------------------------------------------
function slScale_Callback(hObject, eventdata, handles)

val   = get(hObject,'Value'); % steps are 0.02
pos = GetScale(handles);

SetScale(handles, pos + val/4, true)
set(hObject,'Value',0);

% --------------------------------------------------------------------
function pbModify_Callback(hObject, ~, handles)

Rotate_buttons = [handles.pbRotateXY, handles.pbRotateXZ, handles.pbRotateYZ];
Translate_buttons = [handles.pbTranslateX, handles.pbTranslateY, handles.pbTranslateZ];
Scale_buttons = [handles.pbScaleX, handles.pbScaleY, handles.pbScaleZ, handles.pbScaleAll];
Mirror_buttons = [handles.pbMirrorNo, handles.pbMirrorX, handles.pbMirrorY, handles.pbMirrorZ];

switch hObject
  case handles.pbRotateXY
    set(Rotate_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.RotateMode = 'XY';
    SetRotation(handles, GetRotation(handles), false);
  case handles.pbRotateXZ
    set(Rotate_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.RotateMode = 'XZ';
    SetRotation(handles, GetRotation(handles), false);
  case handles.pbRotateYZ
    set(Rotate_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.RotateMode = 'YZ';
    SetRotation(handles, GetRotation(handles), false);
  case handles.pbTranslateX
    set(Translate_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.TranslateMode = 'X';
    SetTranslation(handles, GetTranslation(handles), false);
  case handles.pbTranslateY
    set(Translate_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.TranslateMode = 'Y';
    SetTranslation(handles, GetTranslation(handles), false);
  case handles.pbTranslateZ
    set(Translate_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.TranslateMode = 'Z';
    SetTranslation(handles, GetTranslation(handles), false);
  case handles.pbScaleAll
    set(Scale_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.ScaleMode = 'a';
    SetScale(handles, GetScale(handles), false);
  case handles.pbScaleX
    set(Scale_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.ScaleMode = 'x';
    SetScale(handles, GetScale(handles), false);
  case handles.pbScaleY
    set(Scale_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.ScaleMode = 'y';
    SetScale(handles, GetScale(handles), false);
  case handles.pbScaleZ
    set(Scale_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.ScaleMode = 'z';
    SetScale(handles, GetScale(handles), false);
  case handles.pbMirrorNo
    set(Mirror_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.MirrorMode = '-';
    SetMirror(handles, true);
  case handles.pbMirrorX
    set(Mirror_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.MirrorMode = 'X';
    SetMirror(handles, true);
  case handles.pbMirrorY
    set(Mirror_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.MirrorMode = 'Y';
    SetMirror(handles, true);
  case handles.pbMirrorZ
    set(Mirror_buttons, 'Value', 0);
    set(hObject, 'Value', 1);
    handles.MirrorMode = 'Z';
    SetMirror(handles, true);    
end

% --------------------------------------------------------------------
function pbFixChange_Callback(hObject, eventdata, handles)
arbuz_ApplyTransformation(handles.hh, '', 'FIX');
Reset_sliders(handles);
set([handles.pbMirrorX, handles.pbMirrorY, handles.pbMirrorZ], 'Value', 0);
set(handles.pbMirrorNo, 'Value', 1);

% --------------------------------------------------------------------
function pbReset_Callback(hObject, eventdata, handles)
Reset_sliders(handles);
handles = guidata(handles.figure1);
SendTransformation(handles)

% --------------------------------------------------------------------
function Reset_sliders(handles)

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];
handles.Scale     = [1,1,1];

handles.AAA = {};

guidata(handles.figure1, handles);

set(handles.eRotate, 'string', '0');
set(handles.eTranslate, 'string', '0.000');
set(handles.eScale, 'string', '1');

% --------------------------------------------------------------------
function eRotate_Callback(hObject, eventdata, handles)
val   = str2double(get(handles.eRotate,'String'));
SetRotation(handles, val, true);

% --------------------------------------------------------------------
function eTranslate_Callback(hObject, eventdata, handles)
val   = str2double(get(handles.eTranslate,'String'));
SetTranslation(handles, val, true);

function eScale_Callback(hObject, eventdata, handles)
val   = str2double(get(handles.eScale,'String'));
SetScale(handles, val, true);


% --------------------------------------------------------------------
function pbUpdateFrame_Callback(hObject, eventdata, handles)

Coordinates = arbuz_get(handles.hh, 'Coordinates');

str = {'Master frame'};
for ii=1:length(Coordinates)
  str{end+1} = Coordinates{ii}.Name;
end

val = max(get(handles.pmReferenceFrame, 'Value'), 1);
set(handles.pmReferenceFrame, 'String', str, 'Value', min(val, length(str)));

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----- S E R V I C E     F U N C T I O N S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function A = EmptyTransformation
A.Type = '?';
A.A = eye(4);
% --------------------------------------------------------------------
function varargout = SetRotation(handles, val, redraw)

rotate = handles.Rotate;

A = EmptyTransformation;
switch handles.RotateMode
  case 'XY', rotate(1) = val; A.A=hmatrix_rotate_z(rotate(1));
    set(handles.eRotate, 'string', num2str(rotate(1))); 
  case 'YZ', rotate(2) = val; A.A=hmatrix_rotate_x(rotate(2));
    set(handles.eRotate, 'string', num2str(rotate(2)));
  case 'XZ', rotate(3) = val; A.A=hmatrix_rotate_y(rotate(3));
    set(handles.eRotate, 'string', num2str(rotate(3)));
end
handles.Rotate = rotate;

A.Type = handles.RotateMode; 
if strcmp(handles.LastTransformation,handles.RotateMode) && ~isempty(handles.AAA)
  handles.AAA{end} = A;
else
  handles.AAA{end+1} = A;
end

handles.LastTransformation = handles.RotateMode;
guidata(handles.figure1, handles);

if redraw
  SendTransformation(handles)
end
if nargout > 0, varargout{1} = handles; end

% --------------------------------------------------------------------
function rotation = GetRotation(handles)

switch handles.RotateMode
  case 'XY', rotation = handles.Rotate(1);
  case 'YZ', rotation = handles.Rotate(2);
  case 'XZ', rotation = handles.Rotate(3);
  otherwise, rotation = 0;
end

% --------------------------------------------------------------------
function varargout = SetTranslation(handles, val, redraw)

trans = handles.Translate;

A = EmptyTransformation;
switch handles.TranslateMode
  case 'X', trans(1) = val; set(handles.eTranslate, 'string', sprintf('%6.3f',trans(1)));
  case 'Y', trans(2) = val; set(handles.eTranslate, 'string', sprintf('%6.3f',trans(2)));
  case 'Z', trans(3) = val; set(handles.eTranslate, 'string', sprintf('%6.3f',trans(3)));
end
handles.Translate = trans;

A.Type = 'TRANSXYZ';
A.A    = hmatrix_translate(trans);
if strcmp(handles.LastTransformation,'TRANSXYZ') && ~isempty(handles.AAA)
  handles.AAA{end} = A;
else
  handles.AAA{end+1} = A;
end

handles.LastTransformation = 'TRANSXYZ';
guidata(handles.figure1, handles);

if redraw
  SendTransformation(handles)
end
if nargout > 0, varargout{1} = handles; end

% --------------------------------------------------------------------
function translation = GetTranslation(handles)

switch handles.TranslateMode
  case 'X', translation = handles.Translate(1);
  case 'Y', translation = handles.Translate(2);
  case 'Z', translation = handles.Translate(3);
  otherwise, translation = 0;
end

% --------------------------------------------------------------------
function varargout = SetScale(handles, val, redraw)

scale = handles.Scale;

switch handles.ScaleMode
  case 'x', scale(1) = val;
    set(handles.eScale, 'string', scale(1));
  case 'y', scale(2) = val; scale(3)=1;
    set(handles.eScale, 'string', scale(2));
  case 'z', scale(3)=val;
    set(handles.eScale, 'string', scale(3));
  case 'a', scale(1) = val; scale(2) = val; scale(3) = val;
    set(handles.eScale, 'string', scale(3));
end
handles.Scale = scale;

A.Type = 'SCALEXYZ';
A.A    = hmatrix_scale(scale);
if strcmp(handles.LastTransformation,'SCALEXYZ') && ~isempty(handles.AAA)
  handles.AAA{end} = A;
else
  handles.AAA{end+1} = A;
end

handles.LastTransformation = 'SCALEXYZ';
guidata(handles.figure1, handles);

if redraw
  SendTransformation(handles)
end
if nargout > 0, varargout{1} = handles; end

% --------------------------------------------------------------------
function scale = GetScale(handles)

switch handles.ScaleMode
  case 'x', scale = handles.Scale(1);
  case 'y',  scale = handles.Scale(2);
  case 'z',  scale = handles.Scale(3);
  case 'a', scale = handles.Scale(1);
  otherwise, scale = 1;
end

% --------------------------------------------------------------------
function varargout = SetMirror(handles, redraw)

A = EmptyTransformation;
clear_rotation = false;
switch handles.MirrorMode
  case 'X', A.A = hmatrix_scale([-1,1,1]);
  case 'Y', A.A = hmatrix_scale([1,-1,1]);
  case 'Z', A.A = hmatrix_scale([1,1,-1]);
  otherwise, A.A = eye(4); clear_rotation = true;
end

if clear_rotation
  if ~isempty(handles.AAA) && (strcmp(handles.AAA{end}.Type, 'X') || strcmp(handles.AAA{end}.Type, 'Y') || strcmp(handles.AAA{end}.Type, 'Z'))
    handles.AAA = handles.AAA(1:end-1);
  end
else
  A.Type = handles.MirrorMode;
  if strcmp(handles.LastTransformation,handles.MirrorMode) && ~isempty(handles.AAA)
    handles.AAA{end} = A;
  else
    handles.AAA{end+1} = A;
  end
end

handles.LastTransformation = handles.MirrorMode;
guidata(handles.figure1, handles);

if redraw
  SendTransformation(handles)
end
if nargout > 0, varargout{1} = handles; end

% --------------------------------------------------------------------
function mirror = GetMirror(handles)

switch handles.MirrorMode
  case 'X', mirror = handles.Rotate(1);
  case 'Y', mirror = handles.Rotate(2);
  case 'Z', mirror = handles.Rotate(3);
  otherwise, mirror = 0;
end

% --------------------------------------------------------------------
function SendTransformation(handles)
find_list = arbuz_FindImage(handles.hh, 'master', 'Selected', 1, {});

AA = eye(4);
for ii=1:length(handles.AAA)
  AA = AA * handles.AAA{ii}.A;
end

nCoord = get(handles.pmReferenceFrame, 'Value');
if nCoord > 1
  Coordinates = arbuz_get(handles.hh, 'Coordinates');
  Acc = Coordinates{nCoord - 1}.A;
  AA = inv(Acc)*AA*Acc;
end

arbuz_SetImage(handles.hh, find_list, 'Aprime', AA);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function pushbutton12_Callback(hObject, ~, handles)
the_message = ['Press [Fix] to store the state. Press [Reset] to return to the stored state.'];
msgbox(sprintf(the_message),'Info', 'help')


% --- Executes on button press in pbSetA.
function pbSetA_Callback(hObject, ~, handles)

AA = EnterADLG;

if ~isempty(AA)
  handles.LastTransformation = [];
  find_list = arbuz_FindImage(handles.hh, 'master', 'Selected', 1, {});
    
  arbuz_SetImage(handles.hh, find_list, 'Aprime', AA);
  arbuz_RedrawAll(handles.hh);
end

