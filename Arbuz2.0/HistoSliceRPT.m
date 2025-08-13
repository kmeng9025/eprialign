function varargout = HistoSliceRPT(varargin)
% HISTOSLICERPT M-file for HistoSliceRPT.fig
% Co-Registration GUI and plug-ins

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2007

% Last Modified by GUIDE v2.5 17-Jun-2016 13:53:56

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @HistoSliceRPT_OpeningFcn, ...
  'gui_OutputFcn',  @HistoSliceRPT_OutputFcn, ...
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
function HistoSliceRPT_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
handles.hh = varargin{1};
handles.options.SliceDir = 'Z';
handles.options.slices = {};
handles.options.images = {};

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = HistoSliceRPT_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function ForceRedraw(h)

handles = guidata(h);

if get(handles.cbRedraw, 'Value')
  pbShow_Callback([], [], handles);
end

% --------------------------------------------------------------------
function pbShow_Callback(hObject, eventdata, handles)

handles.options.bounds1 = eval(get(handles.eBoundsX, 'String'));
handles.options.bounds2 = eval(get(handles.eBoundsY, 'String'));
handles.options.FigN = str2double(get(handles.eFigureN, 'String'));

mini = get(handles.ePlaneMin, 'string');
maxi = get(handles.ePlainMax, 'string');

scale = [str2double(mini),str2double(maxi)];
if any(isnan(scale)), scale = []; end

for ii=1:length(handles.options.slices)
	handles.options.slices{ii}.scale = scale;
end

tic
arbuz_HistoSliceRDR(handles.hh, handles.options)
toc

guidata(handles.figure1, handles);
% --------------------------------------------------------------------
function cache = move_to_cache(im, x, y, data, AA)

cache.x = x;
cache.y = y;
cache.im = data;
cache.AA = AA;
cache.ImageIdx = im.ImageIdx;
cache.ProxyIdx = im.ProxyIdx;

% --------------------------------------------------------------------
function ret = is_nochange(im1, im2)

ret = 1;
if isempty(im1) || isempty(im2) || ...
    im1.ImageIdx ~= im2.ImageIdx || im1.ProxyIdx ~= im2.ProxyIdx || ...
    ~isequal(im1.AA, im2.AA)
  ret = 0;
end

% --------------------------------------------------------------------
function pbBringUp_Callback(hObject, eventdata, handles)
FigN = str2double(get(handles.eFigureN, 'String'));
figure(FigN);

% --------------------------------------------------------------------
function pbUpdateImageList_Callback(hObject, eventdata, handles)
[~, str] = GetImageList(handles);

set(handles.pmReferenceFrame, 'string', str);

% --------------------------------------------------------------------
function listbox1_Callback(hObject, eventdata, handles)
sel = get(handles.listbox1, 'value');
if sel < 0, return; end;

alpha = safeget(handles.options.images{sel}, 'alpha', 0.5);
set(handles.eAlpha, 'string', num2str(alpha));
the_mask = safeget(handles.options.images{sel}, 'mask', '');
set(handles.eMask, 'string', the_mask);
scale = safeget(handles.options.images{sel}, 'scale', [0,1]);
set(handles.eImageMin, 'string', num2str(scale(1)));
set(handles.eImageMax, 'string', num2str(scale(2)));

the_colormap = safeget(handles.options.images{sel}, 'colormap', 'jet');
str = get(handles.pmColormap, 'string');
for ii=1:length(str)
  if strcmp(str{ii}, the_colormap)
    set(handles.pmColormap, 'value', ii);
    break;
  end
end
the_colorbar = safeget(handles.options.images{sel}, 'colorbar', 1);
set(handles.pmColorbar, 'Value', the_colorbar);

the_contour = safeget(handles.options.images{sel}, 'contour', '');
set(handles.eContour, 'string', the_contour);
contour_level = safeget(handles.options.images{sel}, 'contour_level', [0.5,0,0]);
contour_index = safeget(handles.options.images{sel}, 'contour_index', [true,false,false]);
contour_controls = [handles.eContourLevel1, handles.eContourLevel2, handles.eContourLevel3];
contour_color_controls = [handles.pbContourColor1, handles.pbContourColor2, handles.pbContourColor3];
contour_color = safeget(handles.options.images{sel},'contour_colors', [1,1,1;1,1,1;1,1,1]);
for ii=1:3
  if contour_index(ii)
    set(contour_controls(ii), 'string', num2str(contour_level(ii)));
  else
    set(contour_controls(ii), 'string', '');
  end
  set(contour_color_controls(ii), 'BackgroundColor', contour_color(ii,:));
end
the_linewidth = safeget(handles.options.images{sel}, 'contour_linewidth', 0.5);
set(handles.eLinewidth, 'string', num2str(the_linewidth));
how_to_mask = safeget(handles.options.images{sel}, 'how_to_mask', 1);
set(handles.pmHowToMask, 'Value', how_to_mask);


% --------------------------------------------------------------------
function pbAdd_Callback(hObject, eventdata, handles)
[~, str] = GetImageList(handles);
[sel,isOk] = listdlg('PromptString','Select an image:',...
  'SelectionMode','single',...
  'ListString',str);
if isOk
  handles.options.images{end+1}.name = str{sel};
  UpdateImageList(handles);
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function UpdateImageList(handles)
str = {};
for ii=1:length(handles.options.images)
  str{end+1} = handles.options.images{ii}.name;
end
val = get(handles.listbox1, 'value');
if isempty(str)
  set(handles.listbox1, 'string', {'Empty list'}, 'value', 1);
else
  set(handles.listbox1, 'string', str, 'value', min(length(str), val));
end

% --------------------------------------------------------------------
function UpdateSliceList(handles)
str = {};
for ii=1:length(handles.options.slices)
  str{end+1} = handles.options.slices{ii}.name;
end
val = get(handles.lbHistoSlices, 'value');
if isempty(str)
  set(handles.lbHistoSlices, 'string', {'Empty list'}, 'value', 1);
else
  set(handles.lbHistoSlices, 'string', str, 'value', min(length(str), val));
end

% --------------------------------------------------------------------
function pbRemove_Callback(hObject, eventdata, handles)
sel = get(handles.listbox1, 'value');
if sel < 0, return; end;
handles.options.images = handles.options.images([1:sel-1,sel+1:end]);
UpdateImageList(handles);
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function ImageData_Callback(hObject, eventdata, handles)
sel = get(handles.listbox1, 'value');
if sel < 0, return; end;

alpha = eval(get(handles.eAlpha, 'string'));
handles.options.images{sel}.alpha = alpha;
handles.options.images{sel}.mask = get(handles.eMask, 'string');
handles.options.images{sel}.scale(1) = str2double(get(handles.eImageMin, 'string'));
handles.options.images{sel}.scale(2) = str2double(get(handles.eImageMax, 'string'));
str = get(handles.pmColormap, 'String');
val = get(handles.pmColormap, 'Value');
handles.options.images{sel}.colormap = str{val};
handles.options.images{sel}.colorbar = get(handles.pmColorbar, 'Value');

handles.options.images{sel}.contour = strtrim(get(handles.eContour, 'string'));
contour_controls = [handles.eContourLevel1, handles.eContourLevel2, handles.eContourLevel3];
contour_color_controls = [handles.pbContourColor1, handles.pbContourColor2, handles.pbContourColor3];
for ii=1:3
  val = strtrim(get(contour_controls(ii), 'string'));
  if isempty(val)
    handles.options.images{sel}.contour_index(ii) = false;
  else
    handles.options.images{sel}.contour_index(ii) = true;
    handles.options.images{sel}.contour_level(ii) = str2double(val);
  end
  handles.options.images{sel}.contour_colors(ii,:) = ...
    get(contour_color_controls(ii), 'BackgroundColor');
end
handles.options.images{sel}.contour_linewidth = ...
  str2double(get(handles.eLinewidth, 'string'));
handles.options.images{sel}.how_to_mask = get(handles.pmHowToMask, 'Value');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbSelectMask_Callback(hObject, eventdata, handles)
sel = get(handles.listbox1, 'value');
if sel < 0, return; end;

find_master    = arbuz_FindImage(handles.hh, 'master', 'Name', handles.options.images{sel}.name, {'SlaveList'});
find_mask    = arbuz_FindImage(handles.hh, find_master{1}.SlaveList, 'ImageType', '3DMASK', {});

str = {'native', 'minmax', 'minmax_alpha'};
for ii=1:length(find_mask)
  str{end+1}= find_mask{ii}.Slave;
end

[res,isOk] = listdlg('PromptString','Select an image:',...
  'SelectionMode','single', 'ListSize',[160 150],...
  'ListString',str);
if isOk
  set(handles.eMask, 'string', str{res});
  ImageData_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function [find_list3D, str] = GetImageList(handles)
find_list3D    = arbuz_FindImage(handles.hh, 'master', 'ImageType', '3DEPRI', {'SlaveList'});
find_list3DPO2 = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'PO2_pEPRI', {'SlaveList'});
find_list3DAMP = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'AMP_pEPRI', {'SlaveList'});
find_list3DMRI = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'MRI', {'SlaveList'});
find_list3DRAW = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'RAW', {'SlaveList'});
find_list3DAMIRA = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'AMIRA3D', {'SlaveList'});
find_list3DDICOM = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'DICOM3D', {'SlaveList'});

for ii=1:length(find_list3DPO2), find_list3D{end+1} = find_list3DPO2{ii}; end
for ii=1:length(find_list3DAMP), find_list3D{end+1} = find_list3DAMP{ii}; end
for ii=1:length(find_list3DMRI), find_list3D{end+1} = find_list3DMRI{ii}; end
for ii=1:length(find_list3DRAW), find_list3D{end+1} = find_list3DRAW{ii}; end
for ii=1:length(find_list3DAMIRA), find_list3D{end+1} = find_list3DAMIRA{ii}; end
for ii=1:length(find_list3DDICOM), find_list3D{end+1} = find_list3DDICOM{ii}; end

str = {};
for ii=1:length(find_list3D), str{end+1} = find_list3D{ii}.Image; end

[str, idx]=sort(str);
find_list3D = find_list3D(idx);

% for ii=1:length(find_list3D)
%   find_mask    = arbuz_FindImage(handles.hh, find_list3D{ii}.SlaveList, 'ImageType', '3DMASK', {});
%
% end
% --------------------------------------------------------------------
function [find_list2D, str] = GetSliceList(handles)
find_list2D    = arbuz_FindImage(handles.hh, 'master', 'ImageType', '2D', {'SlaveList'});

% for ii=1:length(find_list3DPO2), find_list2D{end+1} = find_list3DPO2{ii}; end

str = {};
for ii=1:length(find_list2D), str{end+1} = find_list2D{ii}.Image; end

[str, idx]=sort(str);
find_list2D = find_list2D(idx);


% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)
handles.options.renderer = 'arbuz_HistoSliceRDR';
str = arbuz_GetSaveParList(handles.hh, handles.options.renderer);
if isempty(str), return; end
[sel,isOk] = listdlg('PromptString','Select an image:',...
  'SelectionMode','single',...
  'ListString',str);
if isOk
  set(handles.figure1, 'name', ['AlphaSliceRPT: ', str{sel}]);
  handles.options = arbuz_GetSavePar(handles.hh, str{sel}, handles.options.renderer);
  UpdateImageList(handles);
  listbox1_Callback(hObject, eventdata, handles);
  
  % update reference frame
  set(handles.eFigureN, 'String', safeget(handles.options, 'FigN', 3));
  
  % This is the bounding box
  set(handles.eBoundsX, 'String',handles.options.boundsX);
  set(handles.eBoundsY, 'String',handles.options.boundsY);
  
  str = {};
  for ii=1:length(handles.options.slices)
    str{end+1} = handles.options.slices{ii}.name;
  end
  set(handles.lbHistoSlices, 'string', str)
   
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function pbSave_Callback(hObject, eventdata, handles)
handles.options.renderer = 'arbuz_HistoSliceRDR';
save_pars = arbuz_GetSaveParList(handles.hh, handles.options.renderer);
if isempty(save_pars)
  answer=inputdlg({'Parameter''s name'}, 'Input the name', 1, {'HistoSlice'});
  if ~isempty(answer)
    arbuz_SetSavePar(handles.hh, answer, handles.options.renderer, handles.options);
  end
else
  save_pars = [{'New name'}, save_pars{:}];
  [sel,isOk] = listdlg('PromptString','Select an image:',...
    'SelectionMode','single',...
    'ListString',save_pars);
  if isOk
    if sel == 1
      new_name=inputdlg({'Parameter''s name'}, 'Input the name', 1, {'AlphaSlice'});
      new_name = new_name{1};
    else
      new_name = save_pars{sel};
    end
    arbuz_SetSavePar(handles.hh, new_name, handles.options.renderer, handles.options);
  end
end

% --------------------------------------------------------------------
function ReferenceFrame_Callback(hObject, eventdata, handles)
% create figure layout
handles.options.FigN = get(handles.eFigureN, 'String');

% This is the bounding box
handles.options.boundsX = get(handles.eBoundsX, 'String');
handles.options.boundsY = get(handles.eBoundsY, 'String');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbSelectContour_Callback(hObject, eventdata, handles)
sel = get(handles.listbox1, 'value');
if sel < 0, return; end;

find_master    = arbuz_FindImage(handles.hh, 'master', 'Name', handles.options.images{sel}.name, {'SlaveList'});
find_mask    = arbuz_FindImage(handles.hh, find_master{1}.SlaveList, 'ImageType', '3DMASK', {});

str = {'self'};
for ii=1:length(find_mask)
  str{end+1}= find_mask{ii}.Slave;
end

[res,isOk] = listdlg('PromptString','Select an image:',...
  'SelectionMode','single','ListSize',[160 150],...
  'ListString',str);
if isOk
  set(handles.eContour, 'string', str{res});
  ImageData_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function pbContourColors_Callback(hObject, eventdata, handles)
contour_color_controls = [handles.pbContourColor1, handles.pbContourColor2, handles.pbContourColor3];
idx = find(contour_color_controls == hObject, 1);
if isempty(idx), return; end
the_color = get(hObject, 'BackgroundColor');
new_color = uisetcolor(the_color);
set(hObject, 'BackgroundColor', new_color);
ImageData_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function pbMinMaxShow_Callback(hObject, eventdata, handles)
sel = get(handles.listbox1, 'value');
if sel < 0, return; end;

image_name = handles.options.images{sel}.name;
res = arbuz_FindImage(handles.hh, 'master', 'Name', image_name, {'data'});
dmin = min(res{1}.data(:));
dmax = max(res{1}.data(:));
set(handles.eImageMin, 'string', num2str(dmin));
set(handles.eImageMax, 'string', num2str(dmax));
handles.options.images{sel}.scale = [dmin, dmax];
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pbPlaneSetMinMax_Callback(hObject, eventdata, handles)
% get number of objects to render
nSlices = length(handles.options.slices);

plist = {'Ashow','data','FullName','Color','Bbox','Mask','SlaveList', 'Anative'};
mini = 1E100;
maxi = -mini;
for ii=1:nSlices
  res = arbuz_FindImage(handles.hh, 'master', 'Name', handles.options.slices{ii}.name, plist);
  mini = min([mini,min(res{1}.data(:))]);
  maxi = max([maxi,max(res{1}.data(:))]);
end
set(handles.ePlaneMin, 'string', num2str(mini))
set(handles.ePlainMax, 'string', num2str(maxi))

% --------------------------------------------------------------------
function pbAddHistoSlice_Callback(hObject, eventdata, handles)
[~, str] = GetSliceList(handles);
[sel,isOk] = listdlg('PromptString','Select an image:',...
  'SelectionMode','multiple',...
  'ListString',str);
if isOk
  for ii=1:length(sel)
    handles.options.slices{end+1}.name = str{sel(ii)};
  end
  UpdateSliceList(handles);
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function pbRemoveHistoSlice_Callback(hObject, eventdata, handles)
sel = get(handles.lbHistoSlices, 'value');
if sel < 0, return; end;
handles.options.slices = handles.options.slices([1:sel-1,sel+1:end]);
UpdateSliceList(handles);
guidata(handles.figure1, handles);

