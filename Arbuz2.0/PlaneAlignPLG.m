function varargout = PlaneAlignPLG(varargin)
% PLANEALIGNPLG MATLAB code for PlaneAlignPLG.fig
%      PLANEALIGNPLG, by itself, creates a new PLANEALIGNPLG or raises the existing
%      singleton*.
%
%      H = PLANEALIGNPLG returns the handle to a new PLANEALIGNPLG or the handle to
%      the existing singleton*.
%
%      PLANEALIGNPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLANEALIGNPLG.M with the given input arguments.
%
%      PLANEALIGNPLG('Property','Value',...) creates a new PLANEALIGNPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PlaneAlignPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PlaneAlignPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help PlaneAlignPLG

% Last Modified by GUIDE v2.5 17-Jun-2016 14:08:59

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PlaneAlignPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @PlaneAlignPLG_OutputFcn, ...
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
function PlaneAlignPLG_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};
handles.find_list2D = {};

handles.TranslateMode = 'X';
handles.RotateMode    = 'XY';

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];

handles.TranslateH = eye(4);
handles.RotateH    = eye(4);

step = 0.1; % mm
slmax = get(handles.slX, 'Max');
slmin = get(handles.slX, 'Min');
set(handles.slX, 'SliderStep', [1,10]*step /(slmax-slmin));
set(handles.slY, 'SliderStep', [1,10]*step /(slmax-slmin));
step = 0.5; % degree
slmax = get(handles.slRotation, 'Max');
slmin = get(handles.slRotation, 'Min');
set(handles.slRotation, 'SliderStep', [1,10]*step /(slmax-slmin));

guidata(hObject, handles);

pbUpdateTransformations_Callback(hObject, eventdata, handles);
pbTransUnlock_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function varargout = PlaneAlignPLG_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --------------------------------------------------------------------
function slRotation_Callback(hObject, eventdata, handles) %#ok<DEFNU>

pos = get(handles.lbImageList, 'value');
handles.find_list2D{pos}.phi = get(handles.slRotation, 'value');

UpdateSliceDisplays(handles, pos);
guidata(hObject, handles);

SetCurrentSliceTransformation(handles, pos);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function eRotation_Callback(hObject, eventdata, handles) %#ok<DEFNU>

pos = get(handles.lbImageList, 'value');
handles.find_list2D{pos}.phi = str2double(get(handles.eRotation, 'string'));
guidata(hObject, handles);

set(handles.slRotation, 'value', handles.find_list2D{pos}.phi);

SetCurrentSliceTransformation(handles, pos);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function slY_Callback(hObject, eventdata, handles) %#ok<DEFNU>

pos = get(handles.lbImageList, 'value');
handles.find_list2D{pos}.y = get(handles.slY, 'value');

UpdateSliceDisplays(handles, pos);
guidata(hObject, handles);

SetCurrentSliceTransformation(handles, pos);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function slX_Callback(hObject, eventdata, handles) %#ok<DEFNU>

pos = get(handles.lbImageList, 'value');
handles.find_list2D{pos}.x = get(handles.slX, 'value');

UpdateSliceDisplays(handles, pos);
guidata(hObject, handles);

SetCurrentSliceTransformation(handles, pos);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function lbImageList_Callback(hObject, ~, handles)

pos = get(handles.lbImageList, 'value');
UpdateRefSlice(handles, pos);
pmRotateCenter_Callback(hObject, [], guidata(handles.figure1));

UpdateSliceDisplays(guidata(handles.figure1), pos);
UpdateSliceSliders(guidata(handles.figure1), pos);

handles = guidata(handles.figure1);
UpdateListbox(handles.lbSliceAnchors, handles.find_list2D{pos}.Ref);

% --------------------------------------------------------------------
function eShift_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbMoveSliceUp_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbMoveSliceDn_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbExcludeSlice_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbUpdateTransformations_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>
% Load list of available transformation
Transformations = arbuz_get(handles.hh, 'Transformations');
str = {};
for ii=1:length(Transformations)
  str{end+1} = Transformations{ii}.Name;
end
set(handles.pmTransformation, 'String', str);
set(handles.pmVolumeTransformation, 'String', str);

% load list of available images
im = arbuz_FindImage(handles.hh, 'master', '', '', {});

str = {};
for ii=1:length(im)
  str{end+1} = im{ii}.Image;
end
set(handles.pmRefImage, 'string', str, 'value', 1);

handles.Ref1 = {};
guidata(hObject, handles);
UpdateListbox(handles.lbRefImage, handles.Ref1);

% --------------------------------------------------------------------
function pmTransformation_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>

% --------------------------------------------------------------------
function eSliceThick_Callback(hObject, eventdata, handles)
zpos = GetZPosition(handles);
trans_name = GetSliceTransformationName(handles);

for n_slice = 1:length(handles.find_list2D)
  handles.find_list2D{n_slice}.z = zpos(n_slice);
  handles.find_list2D{n_slice}.AA = BuildA4Slice(handles.find_list2D{n_slice});
  arbuz_SetTransformation(handles.hh, trans_name, handles.find_list2D{n_slice}.Image, handles.find_list2D{n_slice}.AA);
end

guidata(hObject, handles);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function eBlade_Callback(hObject, eventdata, handles)
eSliceThick_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function pbClearAllSlice_Callback(hObject, eventdata, handles) %#ok<DEFNU>
zpos = GetZPosition(handles);
trans_name = GetSliceTransformationName(handles);

for n_slice = 1:length(handles.find_list2D)
  handles.find_list2D{n_slice}.x = 0;
  handles.find_list2D{n_slice}.y = 0;
  handles.find_list2D{n_slice}.z = zpos(n_slice);
  handles.find_list2D{n_slice}.phi = 0;
  handles.find_list2D{n_slice}.AA = BuildA4Slice(handles.find_list2D{n_slice});
  arbuz_SetTransformation(handles.hh, trans_name, handles.find_list2D{n_slice}.Image, handles.find_list2D{n_slice}.AA);
end

guidata(hObject, handles);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function UpdateList(handles)

str = {};

for ii=1:length(handles.find_list2D)
  str{end+1} = ['Slice ', num2str(ii), ': image ''', handles.find_list2D{ii}.FullName, ''''];
end

set(handles.lbImageList, 'string', str)

% --------------------------------------------------------------------
function LoadAllTransformations(handles)

str = get(handles.pmTransformation, 'String');
trSlice = get(handles.pmTransformation, 'Value');
slice_trans_name = str{trSlice};

str = get(handles.pmVolumeTransformation, 'String');
pos = get(handles.pmVolumeTransformation, 'Value');
vol_trans_name = str{pos};

for ii=1:length(handles.find_list2D)
  A = arbuz_GetTransformation(handles.hh, slice_trans_name, handles.find_list2D{ii}.Image);
  handles.find_list2D{ii}.AA = A;
  handles.find_list2D{ii}.Avol =  ...
    arbuz_GetTransformation(handles.hh, vol_trans_name, handles.find_list2D{ii}.Image);
  z(ii) = A(4,3);
  handles.find_list2D{ii}.Ref = {};
  [handles.find_list2D{ii}.Abefore, a, b ] = ...
    arbuz_GetImageTransformation(handles.hh, handles.find_list2D{ii}.Image, ...
  arbuz_get(handles.hh, 'ActiveSequence'),...  
  trSlice, arbuz_get(handles.hh, 'WatchTransformation'));
  handles.find_list2D{ii}.Acenter = eye(4);
end

set(handles.eSliceThick, 'String', sprintf('%g,', z)); 

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pmRotateCenter_Callback(hObject, eventdata, handles)
pos = get(handles.lbImageList, 'value');
ref = get(handles.pmRotateCenter, 'value');

if ref == 1
  Acenter = eye(4); 
else
  Acenter = hmatrix_translate(handles.find_list2D{pos}.xyz{ref-1}.data);
end

A = handles.find_list2D{pos}.AA;
A(2,2) = A(1,1); % Remove flip
A(2,1) = -A(1,2); % Remove flip
handles.find_list2D{pos}.A = eye(4);
handles.find_list2D{pos}.Acenter = Acenter;
handles.find_list2D{pos}.x = A(4,1);
handles.find_list2D{pos}.y = A(4,2);
handles.find_list2D{pos}.z = A(4,3);
handles.find_list2D{pos}.phi = get_angle(A);
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function UpdateRefSlice(handles, n_slice)
if n_slice > length(handles.find_list2D), return; end
slice = handles.find_list2D{n_slice};
find_xyz = arbuz_FindImage(handles.hh, slice, '', '', {'SlaveList'});
handles.find_list2D{n_slice}.xyz = arbuz_FindImage(handles.hh, find_xyz{1}.SlaveList, 'ImageType', 'XYZ', {'FullName', 'data'});
str = {'Image center'};
for ii=1:length(handles.find_list2D{n_slice}.xyz), str{end+1} = handles.find_list2D{n_slice}.xyz{ii}.FullName; end
set(handles.pmRotateCenter, 'string', str, 'value', 1);
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function UpdateSliceDisplays(handles, n_slice)

if n_slice > length(handles.find_list2D), return; end
slice = handles.find_list2D{n_slice};

set(handles.eShift, 'string', sprintf('[%5.2g,%5.2g]', slice.x, slice.y))
set(handles.eRotation, 'string', sprintf('%6.3g', slice.phi))
set(handles.eZ, 'string', sprintf('%6.3g', slice.z))

% --------------------------------------------------------------------
function UpdateSliceSliders(handles, n_slice)

if n_slice > length(handles.find_list2D), return; end
slice = handles.find_list2D{n_slice};
set(handles.slX, 'value', slice.x);
set(handles.slY, 'value', slice.y);
set(handles.slRotation, 'value', slice.phi);

% --------------------------------------------------------------------
function trans_name = GetSliceTransformationName(handles)
str = get(handles.pmTransformation, 'String');
pos = get(handles.pmTransformation, 'Value');
trans_name = str{pos};

% --------------------------------------------------------------------
function SetCurrentSliceTransformation(handles, n_slice)

slice = handles.find_list2D{n_slice};
trans_name = GetSliceTransformationName(handles);

zpos = GetZPosition(handles);

slice.z = zpos(n_slice);
A = BuildA4Slice(slice);

handles.find_list2D{n_slice}.AA = A;
guidata(handles.figure1, handles);

arbuz_SetTransformation(handles.hh, trans_name, slice.Image, A);

% --------------------------------------------------------------------
function the_angle = get_angle(matrix)

the_angle = atan2(matrix(1,2),matrix(1,1))*180/pi;

% --------------------------------------------------------------------
function zpos = GetZPosition(handles)

% Determine positions of the edges
zpos = str2num(get(handles.eSliceThick, 'String')); %#ok<ST2NM>
zpos(20) = 0;

% --------------------------------------------------------------------
function slRotateVolume_Callback(hObject, eventdata, handles)

val   = get(hObject,'Value'); % steps are 0.02
pos = GetRotation(handles);

SetRotation(handles, pos + val*50, true);
set(hObject,'Value',0);

% --------------------------------------------------------------------
function slTranslateVolume_Callback(hObject, eventdata, handles)

str  = get(handles.pmTranslateStep, 'String');
val  = get(handles.pmTranslateStep, 'Value');
step = str2double(str{val});

val   = get(hObject,'Value');
pos = GetTranslation(handles);

SetTranslation(handles, pos + step*val, true);
set(hObject,'Value',0);

% --------------------------------------------------------------------
function eRotateAll_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function eTranslateAll_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function varargout = SetRotation(handles, val, redraw)

rotate = handles.Rotate;

switch handles.RotateMode
  case 'XY', rotate(1) = val; handles.RotateH=hmatrix_rotate_z(rotate(1));
    set(handles.eRotateAll, 'string', num2str(rotate(1)));
  case 'YZ', rotate(2) = val; handles.RotateH=hmatrix_rotate_x(rotate(2));
    set(handles.eRotateAll, 'string', num2str(rotate(2)));
  case 'XZ', rotate(3) = val; handles.RotateH=hmatrix_rotate_y(rotate(3));
    set(handles.eRotateAll, 'string', num2str(rotate(3)));
end
handles.Rotate = rotate;
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

switch handles.TranslateMode
  case 'X', trans(1) = val; 
  case 'Y', trans(2) = val; 
  case 'Z', trans(3) = val; 
end
set(handles.eTranslateAll, 'string', sprintf('[%6.3f, %6.3f, %6.3f]', trans(1), trans(2), trans(3)));
handles.Translate = trans;
handles.TranslateH = hmatrix_translate(trans);
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
function SendTransformation(handles)

str = get(handles.pmVolumeTransformation, 'String');
pos = get(handles.pmVolumeTransformation, 'Value');
trans_name = str{pos};

% if get(handles.pmReferenceFrame, 'Value') > 1
%   hhandles = guidata(handles.hh);
%   Acc = hhandles.Coordinates{1}.A;
%   AA = inv(Acc)*AA*Acc;
% end

AA =  BuildA4Volume(handles.RotateH,handles.TranslateH);

for ii=1:length(handles.find_list2D)
  slice = handles.find_list2D{ii};
  arbuz_SetTransformation(handles.hh, trans_name, slice.Image, slice.Avol*AA);
end

arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function A = BuildA4Slice(slice)
% use_flip = false;
% slice.Acenter = eye(4);
% slice.Acenter = inv(slice.Acenter);
if slice.flip, FlipY = hmatrix_rotate_x(180); else FlipY = eye(4); end
A = slice.A*FlipY*...
  inv(slice.Acenter)*hmatrix_rotate_z(slice.phi)*slice.Acenter*hmatrix_translate([slice.x, slice.y, slice.z]);

% --------------------------------------------------------------------
function A = BuildA4Volume(RotateH, TranslateH)
A = RotateH*TranslateH;

% --------------------------------------------------------------------
function pbModify_Callback(hObject, eventdata, handles) %#ok<DEFNU>

Rotate_buttons = [handles.pbRotateXY, handles.pbRotateXZ, handles.pbRotateYZ];
Translate_buttons = [handles.pbTranslateX, handles.pbTranslateY, handles.pbTranslateZ];

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
end

% --------------------------------------------------------------------
function pushbutton9_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbTransLock_Callback(hObject, eventdata, handles) %#ok<DEFNU>
handles.isLocked = true;

set([handles.pmVolumeTransformation, handles.pmTransformation, handles.pbTransLock, handles.pbUpdateTransformations], 'Enable', 'off');
set([handles.lbImageList, handles.pbTransUnlock], 'Enable', 'on');
set([handles.pbRotateXY, handles.pbRotateXZ, handles.pbRotateYZ], 'Enable', 'on');
set([handles.pbTranslateX, handles.pbTranslateY, handles.pbTranslateZ], 'Enable', 'on');
set([handles.slRotateVolume, handles.slTranslateVolume, handles.slX, handles.slY, handles.slRotation], 'Enable', 'on');
set([handles.pbLoadAllAnchors, handles.pbClearAllSlice], 'Enable', 'on');

% Load complete list of 2D files
plist = {'FullName'};
handles.find_list2D    = arbuz_FindImage(handles.hh, 'master', 'ImageType', '2D', plist);

for ii=1:length(handles.find_list2D)
  handles.find_list2D{ii}.A = eye(4);
  handles.find_list2D{ii}.phi = 0;
  handles.find_list2D{ii}.x = 0;
  handles.find_list2D{ii}.y = 0;
  handles.find_list2D{ii}.z = 0;
  handles.find_list2D{ii}.flip = mod(ii,2) == 0;
end

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];
handles.TranslateH = eye(4);
handles.RotateH    = eye(4);

guidata(hObject, handles);

LoadAllTransformations(handles);
UpdateList(handles);

pos = get(handles.pmTransformation, 'Value');
UpdateSliceDisplays(handles, pos);
UpdateSliceSliders(handles, pos);
UpdateVolumeControls(handles);

arbuz_RedrawAll(handles.hh);
lbImageList_Callback(hObject, eventdata, guidata(hObject));

% --------------------------------------------------------------------
function pbTransUnlock_Callback(hObject, eventdata, handles)
handles.isLocked = false;

set([handles.pmVolumeTransformation, handles.pmTransformation, handles.pbTransLock, handles.pbUpdateTransformations], 'Enable', 'on');
set([handles.lbImageList, handles.pbTransUnlock], 'Enable', 'off');
set([handles.pbRotateXY, handles.pbRotateXZ, handles.pbRotateYZ], 'Enable', 'off');
set([handles.pbTranslateX, handles.pbTranslateY, handles.pbTranslateZ], 'Enable', 'off');
set([handles.slRotateVolume, handles.slTranslateVolume, handles.slX, handles.slY, handles.slRotation], 'Enable', 'off');
set([handles.pbLoadAllAnchors, handles.pbClearAllSlice], 'Enable', 'off');

% --------------------------------------------------------------------
function pbFixVolumeTransformation_Callback(hObject, eventdata, handles) %#ok<DEFNU>
str = get(handles.pmVolumeTransformation, 'String');
pos = get(handles.pmVolumeTransformation, 'Value');
trans_name = str{pos};

AA =  BuildA4Volume(handles.RotateH,handles.TranslateH);
for ii=1:length(handles.find_list2D)
  slice = handles.find_list2D{ii};
  slice.Avol = slice.Avol * AA;
  arbuz_SetTransformation(handles.hh, trans_name, slice.Image, slice.Avol);
  handles.find_list2D{ii} = slice;  
end

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];
handles.TranslateH = eye(4);
handles.RotateH    = eye(4);

guidata(hObject, handles);

UpdateVolumeControls(handles);

% --------------------------------------------------------------------
function pbResetVolumeTransformation_Callback(hObject, eventdata, handles)

str = get(handles.pmVolumeTransformation, 'String');
pos = get(handles.pmVolumeTransformation, 'Value');
trans_name = str{pos};

for ii=1:length(handles.find_list2D)
  slice = handles.find_list2D{ii};
  arbuz_SetTransformation(handles.hh, trans_name, slice.Image, slice.Avol);
end

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];
handles.TranslateH = eye(4);
handles.RotateH    = eye(4);

guidata(hObject, handles);

UpdateVolumeControls(handles);

% --------------------------------------------------------------------
function UpdateVolumeControls(handles)

set(handles.eTranslateAll, 'string', sprintf('[%6.3f, %6.3f, %6.3f]', ...
  handles.Translate(1), handles.Translate(2), handles.Translate(3)));
switch handles.RotateMode
  case 'XY', set(handles.eRotateAll, 'string', num2str(handles.Rotate(1)));
  case 'YZ', set(handles.eRotateAll, 'string', num2str(handles.Rotate(2)));
  case 'XZ', set(handles.eRotateAll, 'string', num2str(handles.Rotate(3)));
end

set(handles.slRotateVolume,'Value',0);
set(handles.slTranslateVolume,'Value',0);

% --------------------------------------------------------------------
function pbClearVolume_Callback(hObject, eventdata, handles) %#ok<DEFNU>
str = get(handles.pmVolumeTransformation, 'String');
pos = get(handles.pmVolumeTransformation, 'Value');
trans_name = str{pos};

for ii=1:length(handles.find_list2D)
  slice = handles.find_list2D{ii};
  slice.Avol = eye(4);
  arbuz_SetTransformation(handles.hh, trans_name, slice.Image, slice.Avol);
  handles.find_list2D{ii} = slice;
end

handles.Translate = [0,0,0];
handles.Rotate    = [0,0,0];
handles.TranslateH = eye(4);
handles.RotateH    = eye(4);

guidata(hObject, handles);

UpdateVolumeControls(handles);

arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function pbInfo_Callback(hObject, eventdata, handles)
the_message = ['This GUI controls histological slice transformations.\n\n',...
'1. Select [Slices] and [Volume] transformations. Press [Lock] button.\n',...
'2. Aligh slices. Use Center and fiducials achors for guidance.\n',...
'3. Rotate volume to match anatomic image (MRI/CT).'];
msgbox(sprintf(the_message),'Info', 'help')

% This GUI controls histological slice transformations.   
%  1.Select slices and global transformations.
%  Press Lock button.
% 2. Aligh slices. Use Center and fiducials achors for guidance.
% 3. Rotate volume to match anatomic image (MRI/CT) 


% --------------------------------------------------------------------
function lbSliceAnchors_Callback(hObject, eventdata, handles)
persistent chk2
if isempty(chk2)
      chk2 = 1;
      pause(0.5); %Add a delay to distinguish single click from a double click
      if chk2 == 1
%           fprintf(1,'\nI am doing a single-click.\n\n');
          chk2 = [];
      end
else
      chk2 = [];
      pbOptionsSliceAnchor_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function pbLoadAllAnchors_Callback(hObject, eventdata, handles)

for ii=1:length(handles.find_list2D)
  im = arbuz_FindImage(handles.hh, handles.find_list2D{ii}, '', '', {'SlaveList'});
  xyz = arbuz_FindImage(handles.hh, im{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Ashow', 'Tag1'});
  
  %   arbuz_SetImage(handles.hh, xyz{ii}, 'Tag1', num2str(obj_ref));
  handles.find_list2D{ii}.Ref = {};
  for jj=1:length(xyz)
    if isempty(xyz{jj}.Tag1)
      a = regexp(xyz{jj}.Slave, 'XYZ(?<tag>\d*)', 'names');
      if ~isempty(a)
        obj_ref = fix(str2double(a.tag));
      else
        obj_ref = 1;
      end
    arbuz_SetImage(handles.hh, xyz{jj}, 'Tag1', num2str(obj_ref));
    else
      obj_ref = fix(str2double(xyz{jj}.Tag1));
    end
    handles.find_list2D{ii}.Ref{end+1} = struct('Name', xyz{jj}.Slave, ...
      'ImageIdx', xyz{jj}.ImageIdx, 'SlaveIdx', xyz{jj}.SlaveIdx, ...
      'Data', xyz{jj}.data, 'Ashow', xyz{jj}.Ashow, 'Assign', obj_ref);
  end
end
guidata(hObject, handles);

pos = get(handles.lbImageList, 'value');
UpdateListbox(handles.lbSliceAnchors, handles.find_list2D{pos}.Ref);

% --------------------------------------------------------------------
function pbAddSliceAnchor_Callback(hObject, eventdata, handles)
pos = get(handles.lbImageList, 'value');

im = arbuz_FindImage(handles.hh, handles.find_list2D{pos}, '', '', {'SlaveList'});
if isempty(im), return; end
xyz = arbuz_FindImage(handles.hh, im{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Ashow', 'Tag1'});

str = {};
for ii=1:length(xyz)
  str{end+1} = xyz{ii}.Slave;
end

[sel, ok] = listdlg('ListString', str, 'SelectionMode', 'multiple', 'Name', 'Select anchors');

if ok
  for ii=1:length(sel)
    if isempty(xyz{ii}.Tag1)
      obj_ref = 1;
    else
      obj_ref = fix(str2double(xyz{ii}.Tag1));
    end
    arbuz_SetImage(handles.hh, xyz{ii}, 'Tag1', num2str(obj_ref));
    handles.find_list2D{pos}.Ref{end+1} = struct('Name', str{sel(ii)}, ...
    'ImageIdx', xyz{ii}.ImageIdx, 'SlaveIdx', xyz{ii}.SlaveIdx, ...  
    'Data', xyz{ii}.data, 'Ashow', xyz{ii}.Ashow, 'Assign', obj_ref);
  end
  
  guidata(hObject, handles);
  UpdateListbox(handles.lbSliceAnchors, handles.find_list2D{pos}.Ref);
end

% --------------------------------------------------------------------
function pbRemoveSliceAnchor_Callback(hObject, eventdata, handles)
pos = get(handles.lbImageList, 'value');
val = get(handles.lbSliceAnchors, 'value');

handles.find_list2D{pos}.Ref = handles.find_list2D{pos}.Ref([1:val-1, val+1:end]);
guidata(hObject, handles);
UpdateListbox(handles.lbSliceAnchors, handles.find_list2D{pos}.Ref);

% --------------------------------------------------------------------
function pbOptionsSliceAnchor_Callback(hObject, eventdata, handles)
pos = get(handles.lbImageList, 'value');
val = get(handles.lbSliceAnchors, 'value');

anchor = handles.find_list2D{pos}.Ref{val};
res=inputdlg({'Assignment'}, ['Anchor ', anchor.Name], 1, {num2str(anchor.Assign)});

if ~isempty(res)
  handles.find_list2D{pos}.Ref{val}.Assign = fix(str2double(res{1}));
  arbuz_SetImage(handles.hh, {anchor.ImageIdx, anchor.SlaveIdx}, 'Tag1', res{1});
  guidata(hObject, handles);  
  UpdateListbox(handles.lbSliceAnchors, handles.find_list2D{pos}.Ref);
end

% --------------------------------------------------------------------
function pmRefImage_Callback(hObject, eventdata, handles)
handles.Ref1 = {};
guidata(hObject, handles);
UpdateListbox(handles.lbRefImage, handles.Ref1);

% --------------------------------------------------------------------
function lbRefImage_Callback(hObject, eventdata, handles)
persistent chk1
if isempty(chk1)
      chk1 = 1;
      pause(0.5); %Add a delay to distinguish single click from a double click
      if chk1 == 1
%           fprintf(1,'\nI am doing a single-click.\n\n');
          chk1 = [];
      end
else
      chk1 = [];
      pbOptionsRefImage_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function pbAddRefImage_Callback(hObject, eventdata, handles)
str = get(handles.pmRefImage, 'string');
val = get(handles.pmRefImage, 'val');

im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {'SlaveList'});
xyz = arbuz_FindImage(handles.hh, im{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Ashow', 'Tag1'});

str = {};
for ii=1:length(xyz)
  str{end+1} = xyz{ii}.Slave;
end

if ~isempty(str)
[sel, ok] = listdlg('ListString', str, 'SelectionMode', 'multiple', 'Name', 'Select anchors');

if ok
  for ii=1:length(sel)
    if isempty(xyz{ii}.Tag1)
      obj_ref = 1;
    else
      obj_ref = fix(str2double(xyz{ii}.Tag1));
    end
    arbuz_SetImage(handles.hh, xyz{ii}, 'Tag1', num2str(obj_ref));
    handles.Ref1{end+1} = struct('Name', str{sel(ii)}, ...
      'ImageIdx', xyz{ii}.ImageIdx, 'SlaveIdx', xyz{ii}.SlaveIdx, ...
      'Data', xyz{ii}.data, 'Ashow', xyz{ii}.Ashow, 'Assign', obj_ref);
  end
  
  guidata(hObject, handles);
  UpdateListbox(handles.lbRefImage, handles.Ref1);
end
end

% --------------------------------------------------------------------
function pbRemoveRefImage_Callback(hObject, eventdata, handles)
arbuz_ShowMessage(handles.hh, 'Not implemented. Use image selection to clear anchors.');

% --------------------------------------------------------------------
function pbOptionsRefImage_Callback(hObject, eventdata, handles)
val = get(handles.lbRefImage, 'val');

res=inputdlg({'Assignment'}, ['Anchor ', handles.Ref1{val}.Name], 1, {num2str(handles.Ref1{val}.Assign)});

if ~isempty(res)
  handles.Ref1{val}.Assign = fix(str2double(res{1}));
  arbuz_SetImage(handles.hh, {handles.Ref1{val}.ImageIdx, handles.Ref1{val}.SlaveIdx}, 'Tag1', res{1});
  guidata(hObject, handles);  
  UpdateListbox(handles.lbRefImage, handles.Ref1);
end

% --------------------------------------------------------------------
function pnOptimizeAll_Callback(hObject, eventdata, handles)
% hObject    handle to pnOptimizeAll (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of pnOptimizeAll

% --------------------------------------------------------------------
function UpdateListbox(item_handle, data_to_update)

str = {};
for ii=1:length(data_to_update)
  str{end+1} = sprintf('%s (%i)', data_to_update{ii}.Name, data_to_update{ii}.Assign);
end

val = get(item_handle, 'value');
if isempty(val) || val == 0, val = 1; end
set(item_handle, 'string', str, 'value', 1)

% --------------------------------------------------------------------
function pbRegisterSlice_Callback(hObject, eventdata, handles)

% tic
AA = slice_transform(handles);
% toc

pos = get(handles.lbImageList, 'value');
im = arbuz_FindImage(handles.hh, handles.find_list2D{pos} , '', '', {});

arbuz_SetImage(handles.hh, im, 'Aprime', AA);
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
% distance(x0,y0,z0,a,b,c) = sum{[c(yi - y0) - b(z - z0)]^2 +
%                           [a(zi - z0) - c(x - x0)]^2 +
%                           [b(xi - x0) - a(y - y0)]^2}

% --------------------------------------------------------------------
% points is a Nx3 array where N is the number of points
function [r0, a] = get_best_line3D(points)

N = size(points, 1);
r0 = mean(points);
xyz = points - repmat(r0, [N, 1]);
[a,b,V]=svd(xyz,0); %#ok<ASGLU>
a = V(1:3,1)';   

% --------------------------------------------------------------------
function fiducials = get_fiducials(anchors)

% find fiducials
fiducials = [];
for ii=1:10
  points = [];
  for jj=1:length(anchors)
    if anchors{jj}.Assign == ii
      points(end+1, :) = htransform_vectors(anchors{jj}.Ashow, anchors{jj}.Data);
    end
  end
  if ~isempty(points)
    [r0, a] = get_best_line3D(points);
    fiducials(end+1, :) = [r0,a];
  end
end

% --------------------------------------------------------------------
function sorted_anchors = get_anchors(anchors)
sorted_anchors = [];
for ii=1:10
  for jj=1:length(anchors)
    if anchors{jj}.Assign == ii
      sorted_anchors(end+1, :) = [htransform_vectors(anchors{jj}.Ashow, anchors{jj}.Data), ii];
    end
  end
end

% --------------------------------------------------------------------
function transform_list = update_transformations(handles, transform_list)

% str = get(handles.pmTransformation, 'String');
ActiveTransformation = get(handles.pmTransformation, 'Value');

WatchTransformation=arbuz_get(handles.hh, 'WatchTransformation');
% ActiveTransformation

for ii=1:length(transform_list)
 xyz = arbuz_FindImage(handles.hh, {struct('ImageIdx', transform_list{ii}.ImageIdx, 'SlaveIdx', transform_list{ii}.SlaveIdx)}, ...
    '', '', {'data', 'Ashow', 'A', 'Apre', 'Anext', 'Ashow'});
 
 parent = arbuz_FindImage(handles.hh, {transform_list{ii}.ImageIdx}, '', '', {});
  
 [Apre, Acurrent, Apost] = arbuz_GetImageTransformation(handles.hh, parent{1}.Image, 1, ActiveTransformation, WatchTransformation);
 
 transform_list{ii}.data = xyz{1}.data;
 transform_list{ii}.Abefore = Apre*Acurrent;
 transform_list{ii}.Aafter  = Apost;
 transform_list{ii}.Ashow = xyz{1}.Ashow;
end

% --------------------------------------------------------------------
function A = slice_transform(handles)

pos = get(handles.lbImageList, 'value');

handles.Ref1 = update_transformations(handles, handles.Ref1);
handles.find_list2D{pos}.Ref = update_transformations(handles, handles.find_list2D{pos}.Ref);

% find fiducials
fiducials = get_fiducials(handles.Ref1);

% find points
anchors = get_anchors(handles.find_list2D{pos}.Ref);

% here Pxyz any point and RSxyz = [R0, A0] - line definition, R0 any
% point, A0 parameteric slope constants xyz = R0 + A0*t
% distance = @(Pxyz, RSxyz) norm(cross(Pxyz - RSxyz(1:3), RSxyz(4:6))) / norm(RSxyz(4:6));

% form a fitting array
fit_array = anchors(:,1:3);
for ii = 1:size(anchors, 1)
  fit_array(ii, 4:9) = fiducials(anchors(ii,4),:);
end

Abefore = handles.find_list2D{pos}.Ref{2}.Abefore*BuildA4Slice(handles.find_list2D{pos});
Aafter = handles.find_list2D{pos}.Ref{2}.Aafter;

options = optimset('TolFun',1e-4);
x = fminsearch(@fit_func_slice, [0,0,0], options, fit_array, Abefore, Aafter);
% disp(x)
disp(sprintf('Fit error: %5.3f', fit_func_slice(x, fit_array, Abefore, Aafter)));

FlipY = eye(4);
A = FlipY*hmatrix_rotate_z(x(1))*hmatrix_translate([x(2:3), 0]);

anchors1 = get_anchors(handles.Ref1);
figure(100); clf;
for ii=1:max(anchors(:, 4))
  idx = anchors(:, 4) == ii;
  plot3(anchors(idx, 1), anchors(idx, 2), anchors(idx, 3), 'o-b'); hold on;
end
for ii=1:max(anchors1(:, 4))
  idx = anchors1(:, 4) == ii;
  plot3(anchors1(idx, 1), anchors1(idx, 2), anchors1(idx, 3), 'o-r'); hold on;
end
grid on

% --------------------------------------------------------------------
function error = fit_func_slice(x, fit_array, Abefore, Aafter)
FlipY = eye(4);
AA = FlipY*hmatrix_rotate_z(x(1))*hmatrix_translate([x(2:3), 0]);

dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(Abefore*AA*Aafter, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function res = point_to_line_distance(FA) 

res = norm(cross(FA(1:3) - FA(4:6), FA(7:9))) / norm(FA(7:9));

% --------------------------------------------------------------------
function pbAssignAnchors_Callback(hObject, eventdata, handles)
for ii = 1:length(handles.Ref1)
  a = regexp(handles.Ref1{ii}.Name, 'XYZ(?<tag>\d*)', 'names');
  if ~isempty(a) && isstruct(a)
    new_tag = fix(str2double(a.tag));
    handles.Ref1{ii}.Assign = new_tag;
    arbuz_SetImage(handles.hh, {handles.Ref1{ii}.ImageIdx, handles.Ref1{ii}.SlaveIdx}, 'Tag1', num2str(new_tag));
  end
end
UpdateListbox(handles.lbRefImage, handles.Ref1);
