function varargout = ImageMathPLG(varargin)
% IMAGEMATHPLG M-file for ImageMathPLG.fig
%      IMAGEMATHPLG, by itself, creates a new IMAGEMATHPLG or raises the existing
%      singleton*.
%
%      H = IMAGEMATHPLG returns the handle to a new IMAGEMATHPLG or the handle to
%      the existing singleton*.
%
%      IMAGEMATHPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMAGEMATHPLG.M with the given input arguments.
%
%      IMAGEMATHPLG('Property','Value',...) creates a new IMAGEMATHPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ImageMathPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ImageMathPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ImageMathPLG

% Last Modified by GUIDE v2.5 09-Jun-2015 18:09:30

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ImageMathPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @ImageMathPLG_OutputFcn, ...
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
function ImageMathPLG_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<INUSL>
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};

plist = {'FullName'};
find_list3D    = arbuz_FindImage(handles.hh, 'master', 'ImageType', '3DEPRI', plist);
find_list3DPO2 = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'PO2_pEPRI', plist);
find_list3DAMP = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'AMP_pEPRI', plist);
find_list3DMRI = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'MRI', plist);
find_list3DAMIRA = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'AMIRA3D', plist);
find_list3DDICOM = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'DICOM3D', plist);
find_listFITRESULT = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'FITRESULT', plist);

for ii=1:length(find_list3DPO2), find_list3D{end+1} = find_list3DPO2{ii}; end
for ii=1:length(find_list3DAMP), find_list3D{end+1} = find_list3DAMP{ii}; end
for ii=1:length(find_list3DMRI), find_list3D{end+1} = find_list3DMRI{ii}; end
for ii=1:length(find_list3DAMIRA), find_list3D{end+1} = find_list3DAMIRA{ii}; end
for ii=1:length(find_list3DDICOM), find_list3D{end+1} = find_list3DDICOM{ii}; end
for ii=1:length(find_listFITRESULT), find_list3D{end+1} = find_listFITRESULT{ii}; end

handles.find_list = find_list3D;
str = cell(1, length(find_list3D));
for ii=1:length(find_list3D)
  str{ii}=find_list3D{ii}.FullName;
end
set(handles.pmSource, 'String', str);
set(handles.pmSource2, 'String', str);
pos1 = get(handles.pmSource, 'Value');
fill_PopupMenu(handles.pmSourceMask, handles.pmSourceObject, ...
  handles, handles.find_list{pos1});
pos2 = get(handles.pmSource2, 'Value');
fill_PopupMenu(handles.pmSourceMask2, handles.pmSourceObject2, ...
  handles, handles.find_list{pos2});

set(handles.pmDestination, 'String', str);
% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = ImageMathPLG_OutputFcn(hObject, eventdata, handles)  %#ok<INUSL>
% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>
pos1 = get(handles.pmSource, 'Value');
pos2 = get(handles.pmSource2, 'Value');
pos1sl = get(handles.pmSourceMask, 'Value');
pos2sl = get(handles.pmSourceMask2, 'Value');

posDest = get(handles.pmDestination, 'Value');

% get source 3D masks
ProxyList1 = arbuz_FindImage(handles.hh, handles.find_list(pos1), ...
  '', '', {'SlaveList'});
ProxyList2 = arbuz_FindImage(handles.hh, handles.find_list(pos2), ...
  '', '', {'SlaveList'});

switch get(handles.pmSourceObject, 'value')
  case 1
    if pos1sl == 1
    find_image3D = arbuz_FindImage(handles.hh, handles.find_list(pos1), ...
        '', '', {'Ashow', 'data', 'Name'}); 
    op1 = find_image3D{1};
    else
      str = get(handles.pmSourceMask2, 'string');
      find_mask_3D = arbuz_FindImage(handles.hh, ProxyList1{1}.SlaveList, ...
        'Name', str{pos1sl}, {'Ashow','data', 'Name'});
      op1 = find_mask_3D{1};
    end
  case 2
    pos1_mask_1  = get(handles.pmSourceMask2, 'Value');
    find_mask_3D = arbuz_FindImage(handles.hh, ProxyList1{1}.SlaveList, ...
      'ImageType', '3DMASK', {'Ashow', 'data', 'Name'});
    op1 = find_mask_3D{pos1_mask_1};
end

switch get(handles.pmSourceObject2, 'value')
  case 1
    if pos2sl == 1
      find_image3D = arbuz_FindImage(handles.hh, handles.find_list(pos2), ...
        '', '', {'Ashow', 'data', 'Name'});
      op2 = find_image3D{1};
    else
      str = get(handles.pmSourceMask2, 'string');
      find_mask_3D = arbuz_FindImage(handles.hh, ProxyList2{1}.SlaveList, ...
        'Name', str{pos2sl}, {'Ashow','data', 'Name'});
      op2 = find_mask_3D{1};
    end
  case 2
    pos1_mask_2  = get(handles.pmSourceMask2, 'Value');
    find_mask_3D = arbuz_FindImage(handles.hh, ProxyList2{1}.SlaveList, ...
      'ImageType', '3DMASK', {'Ashow', 'data', 'Name'});
    op2 = find_mask_3D{pos1_mask_2};
end

if ~isequal(size(op1.data),size(op2.data))
  set(handles.eLog, 'string', 'Sizes are not =', 'backgroundcolor', [1 0 0]);
  return;
end

% get destination dimensions
find_dest3D = arbuz_FindImage(handles.hh, handles.find_list(posDest), ...
  '', '', {'Ashow','Box'});

%%cut_off = str2double(get(handles.eCutOff, 'String'));
%%dilate_factor = fix(str2double(get(handles.eMaskDilate, 'String')));

set(handles.eLog, 'string', 'Processing...', 'backgroundcolor', [1 0 0]);
drawnow;

new_image = [];
new_image.Name = get(handles.eDestinationMask, 'String');
new_image.isStore = 1;
new_image.ImageType = op1.ImageType;

the_value =  str2double(get(handles.eValue, 'string'));
switch get(handles.pmMath, 'value')
  case 1, new_image.data = double(op1.data) + (double(op2.data)*the_value);    
  case 2, new_image.data = double(op1.data) - (double(op2.data)*the_value);    
  case 3, new_image.data = double(op1.data) .* (double(op2.data)*the_value);    
  case 4, new_image.data = double(op1.data) ./ (double(op2.data)*the_value);    
end

arbuz_AddImage(handles.hh, new_image, find_dest3D{1}.ImageIdx);

set(handles.eLog, 'string', 'Finished!', 'backgroundcolor', [0 1 0]);

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pmSource_Callback(hObject, eventdata, handles)
pos1 = get(handles.pmSource, 'Value');
fill_PopupMenu(handles.pmSourceMask, handles.pmSourceObject, ...
  handles, handles.find_list{pos1});

% --------------------------------------------------------------------
function pmSourceObject_Callback(hObject, eventdata, handles)
pos1 = get(handles.pmSource, 'Value');
fill_PopupMenu(handles.pmSourceMask, handles.pmSourceObject, ...
  handles, handles.find_list{pos1});

% --------------------------------------------------------------------
function fill_PopupMenu(pm, pm2, handles, im_name)
ProxyList = arbuz_FindImage(handles.hh, {im_name}, '', '', {'SlaveList'});
find_mask3D = arbuz_FindImage(handles.hh, ProxyList{1}.SlaveList, ...
  'ImageType', '3DMASK', {'Name'});
find_image3D = arbuz_FindImage(handles.hh, ProxyList{1}.SlaveList, ...
  'ImageType', 'MRI', {'Name'});

the_type = get(pm2, 'value');

if(the_type == 2)
    str = cell(length(find_mask3D), 1);
    for ii=1:length(find_mask3D), str{ii}=find_mask3D{ii}.Name; end
    if isempty(str), str = {'None'}; end
else
    str = cell(length(find_image3D)+1, 1);
    str{1} = 'image itself';
    for ii=1:length(find_image3D), str{ii+1}=find_image3D{ii}.Name; end
    if isempty(str), str = {'None'}; end
end

set(pm, 'String', str, 'Value', 1)
%%set(handles.eDestinationMask, 'String', str{1});

set(handles.eLog, 'string', 'Ready.', 'backgroundcolor', [1 1 1]);

% --------------------------------------------------------------------
function pmSourceMask_Callback(hObject, eventdata, handles) %#ok<INUSL,DEFNU>
str = get(handles.pmSourceMask, 'String');
val = get(handles.pmSourceMask, 'Value');
set(handles.eDestinationMask, 'String', str{val});

% --------------------------------------------------------------------
function pmSource2_Callback(hObject, eventdata, handles)
pos2 = get(handles.pmSource2, 'Value');
fill_PopupMenu(handles.pmSourceMask2, handles.pmSourceObject2, ...
  handles, handles.find_list{pos2});

% --------------------------------------------------------------------
function pmSourceObject2_Callback(hObject, eventdata, handles)
pos2 = get(handles.pmSource2, 'Value');
fill_PopupMenu(handles.pmSourceMask2, handles.pmSourceObject2, ...
  handles, handles.find_list{pos2});
