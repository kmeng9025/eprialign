function varargout = TTouchImageBasedPLG(varargin)
% TTOUCHIMAGEBASEDPLG M-file for TTouchImageBasedPLG.fig
%      TTOUCHIMAGEBASEDPLG, by itself, creates a new TTOUCHIMAGEBASEDPLG or
%      raises the existing
%      singleton*.
%
%      H = TTOUCHIMAGEBASEDPLG returns the handle to a new TTOUCHIMAGEBASEDPLG or
%      the handle to
%      the existing singleton*.
%
%      TTOUCHIMAGEBASEDPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TTOUCHIMAGEBASEDPLG.M with the given input arguments.
%
%      TTOUCHIMAGEBASEDPLG('Property','Value',...) creates a new
%      TTOUCHIMAGEBASEDPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before TTouchImageBasedPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property
%      application
%      stop.  All inputs are passed to TTouchImageBasedPLG_OpeningFcn via
%      varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help TTouchImageBasedPLG

% Last Modified by GUIDE v2.5 22-Oct-2018 10:53:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @TTouchImageBasedPLG_OpeningFcn, ...
  'gui_OutputFcn',  @TTouchImageBasedPLG_OutputFcn, ...
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
function TTouchImageBasedPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for Image2ContourPLG
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};
handles.contours = {};
handles.Masks = {};
handles.XYZs = {};
handles.ViewDirection = 3;
handles.ApplyTo = 1;
handles.cursor = [];
handles.op_mode = 'none';
handles.draw_mode = 'none';
handles.rubber_radius = 3;
handles.is2D = 0;
handles.opt.vis_abs = 0;

set(handles.figure1,'KeyPressFcn',@KeyPressFunction);

handles.icons = load('SliceMaskEditICO.mat');

% Update handles structure
guidata(hObject, handles);
handles = BuildEllipseTool(handles);


% try to load selected dataset
pbLoad_Callback(hObject, [], handles)

% --------------------------------------------------------------------
function varargout = TTouchImageBasedPLG_OutputFcn(hObject, eventdata, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)
find_list = GetSelectedImage(handles);

handles.Masks = {};
handles.XYZs = {};
handles.MaskBuffer = [];
handles.is2D = 0;
handles.isSlave = 0;

if ~isempty(find_list)
  im = arbuz_FindImage(handles.hh, {find_list}, '', '', {'data', 'Aslave', 'FullName', 'SlaveList', 'A', 'Box', 'Anative'});
  im = im{1};
  if strcmp(im.ImageType, '2D'), handles.is2D = 1; end
  if strcmp(im.ImageType, '3DSURFACE'), return; end
  if strcmp(im.ImageType, 'XYZ'), return; end
  % im.data(im.data > 100) = 0; % fix this
  % im.data(im.data < -10) = 0; % fix this
  
  if ndims(im.data) > 3
    nnn = size(im.data,4);
    str = cell(nnn,1); for ii=1:nnn, str{ii} = sprintf('%i', ii);end
    res = listdlg('PromptString','Select slice:', 'SelectionMode','single', 'ListString',str);
    im.data = im.data(:,:,:,res);
  end
  
  im.max = cast(max(im.data(:)), 'double');
  im.min = cast(min(im.data(:)), 'double');
  set(handles.figure1, 'Name', sprintf('TTouchImageBasedPLG [%s]',im.FullName));
  
  handles.isSlave = im.SlaveIdx > 0;
  
  if handles.isSlave
    im2 = arbuz_FindImage(handles.hh, {struct('ImageIdx',im.ImageIdx)}, '', '', {'Aslave', 'FullName', 'SlaveList'});
    find_mask3D    = arbuz_FindImage(handles.hh, im2{1}.SlaveList, 'ImageType', '3DMASK', {'data', 'Name', 'A', 'Color'});
    find_XYZ3D     = arbuz_FindImage(handles.hh, im2{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Name', 'A', 'Color'});
  else
    find_mask3D    = arbuz_FindImage(handles.hh, im.SlaveList, 'ImageType', '3DMASK', {'data', 'Name', 'A', 'Color'});
    find_XYZ3D    = arbuz_FindImage(handles.hh, im.SlaveList, 'ImageType', 'XYZ', {'data', 'Name', 'A', 'Color'});
  end
  for ii = 1:length(find_mask3D)
    if isequal(size(find_mask3D{ii}.data), size(im.data))
      handles.Masks{end+1}.Mask = find_mask3D{ii}.data;
      handles.Masks{end}.Name = find_mask3D{ii}.Name;
      handles.Masks{end}.Color = find_mask3D{ii}.Color;
    else
      disp(sprintf('Mask ''%s'' has different resolution and was not loaded.', find_mask3D{ii}.Name));
    end
  end
  for ii = 1:length(find_XYZ3D)
    if handles.isSlave
      handles.XYZs{end+1}.data = htransform_vectors(inv(im.Aslave), find_XYZ3D{ii}.data);
    else
      handles.XYZs{end+1}.data = find_XYZ3D{ii}.data;
    end
      handles.XYZs{end}.Name = find_XYZ3D{ii}.Name;
      if handles.is2D, handles.XYZs{end}.data(3) = 1; end
  end   
else
  im = [];
  set(handles.figure1, 'Name', 'TTouchImageBasedPLG');
end

handles.image = im;
handles.SliceN = SetSlice(handles, 0);

% reset zoom parameters
handles.ax = GetImageDims(handles);

handles.OldMasks{1} = handles.Masks;
handles.OldMasks{2} = handles.Masks;
guidata(hObject, handles);

ListAllMasks(handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
if isempty(handles), return; end
fig_size = get(handles.figure1, 'Position');

tool_size = 185;
wborder = 12;
hborder = 6;
w_axborder = 60;
h_axborder = 15;
h_buttonsize = 28;
h_selectorpanel = 80;
h_buttonpanel = 40;

h_navig_panel = 60;
w_navig_button = 100;
slider_width = 15;
h_ellipse_panel = 60;

panel_size = [fig_size(3)-tool_size-wborder, hborder, tool_size, fig_size(4)-2*hborder];
set(handles.panelTools, 'Position',panel_size);
figure_panel_size = fig_size(3)-tool_size-3*wborder-w_axborder;
set(handles.pEllipse, 'Position', ...
  [wborder, fig_size(4)-h_ellipse_panel, ...
  figure_panel_size, ...
  h_ellipse_panel]);

figure_size = figure_panel_size / 3;
set(handles.axes1, 'Position', ...
  [wborder+w_axborder, hborder+h_axborder+h_navig_panel, ...
  figure_size - w_axborder, ...
  fig_size(4)-2*hborder-h_axborder-h_navig_panel]);
set(handles.axes2, 'Position', ...
  [wborder+figure_size+w_axborder, hborder+h_axborder+h_navig_panel, ...
  figure_size - w_axborder, ...
  fig_size(4)-2*hborder-h_axborder-h_navig_panel]);
set(handles.axes3, 'Position', ...
  [wborder+2*figure_size+w_axborder, hborder+h_axborder+h_navig_panel, ...
  figure_size - w_axborder, ...
  fig_size(4)-2*hborder-h_axborder-h_navig_panel]);

w_navig_panel = fig_size(3)-tool_size-2*w_axborder;
navig_panel_offset = (w_navig_panel - (w_navig_button+wborder)*5)/2 + w_axborder;
h_navig_element = h_navig_panel - 3*hborder - slider_width;
pos = [navig_panel_offset + wborder, 2*hborder + slider_width, w_navig_button, h_navig_panel - 3*hborder - slider_width];
set(handles.pbPrevSlice, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder) * 5, 2*hborder + slider_width, w_navig_button, h_navig_element];
set(handles.pbNextSlice, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder) * 2, 2 * hborder+slider_width, w_navig_button, h_navig_element];
set(handles.eSliceN1, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder) * 2+100, 2 * hborder+slider_width, w_navig_button, h_navig_element];
set(handles.eSliceN2, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder) * 2+200, 2 * hborder+slider_width, w_navig_button, h_navig_element];
set(handles.eSliceN3, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder),2*hborder+slider_width-1, w_navig_button, h_navig_element];
set(handles.textSliceN, 'Position', pos);

set(handles.pbLoad, 'Position', ...
  [wborder, panel_size(4) - h_buttonpanel - hborder, ...
  panel_size(3)-2*wborder-5-h_buttonpanel, h_buttonpanel]);

set(handles.pbInfo, 'Position', ...
  [panel_size(3)-wborder-h_buttonpanel, panel_size(4) - h_buttonpanel - hborder, ...
  h_buttonpanel, h_buttonpanel]);

set(handles.pInfo, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - 40, ...
  panel_size(3)-2*wborder, h_selectorpanel]);

set(handles.uipanel7, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - 150, ...
  panel_size(3)-2*wborder, 106]);

set(handles.pCustomTools, 'Position', ...
  [wborder, 2.5*hborder+2*h_buttonsize, ...
  tool_size-2*wborder, 85]);

set(handles.pbDone, 'Position', ...
  [wborder, 1.5*hborder+h_buttonsize, tool_size-2*wborder, h_buttonsize]);

set(handles.pbSaveSurface, 'Position', ...
  [wborder, hborder, tool_size-2*wborder, h_buttonsize]);

% --------------------------------------------------------------------
function pbPrevSlice_Callback(hObject, eventdata, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, -1);
guidata(hObject, handles);
DrawSlice(handles);


% --------------------------------------------------------------------
function pbNextSlice_Callback(hObject, eventdata, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, +1);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function eSliceN1_Callback(hObject, eventdata, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, 0);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mEditUndo_Callback(hObject, eventdata, handles)
handles = RestoreMasks(handles);
ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function StoreMasks(handles)
if isfield(handles, 'Masks')
  handles.OldMasks{1} = handles.OldMasks{2};
  handles.OldMasks{2} = handles.Masks;
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function handles = RestoreMasks(handles)
if isfield(handles, 'OldMasks')
  handles.Masks = handles.OldMasks{1};
  StoreMasks(handles);
end

% --------------------------------------------------------------------
function pbAddMask_Callback(hObject, eventdata, handles)
val = get(handles.pmMaskType, 'Value');
str = get(handles.pmMaskType, 'String');
handles.Masks{end+1} = struct('Name', str{val}, 'Mask', false(size(handles.image.data)), ...
  'Color', []);
StoreMasks(handles);
ListAllMasks(handles);
set(handles.lbMasks, 'Value', length(handles.Masks));

% --------------------------------------------------------------------
function pbDeleteMask_Callback(hObject, eventdata, handles)
val = get(handles.lbMasks, 'Value');
handles.Masks = handles.Masks([1:val-1,val+1:end]);
StoreMasks(handles)
ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbMaskColor_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function DrawPolyContour(handles, contour_mode)
if isempty(handles.Masks), return; end
nMask = get(handles.lbMasks, 'Value');

handles.draw_mode = 'impoly';
guidata(handles.figure1, handles);

try
  h = impoly(handles.axes1, []);
catch e
  return;
end

api = iptgetapi(h);
if isempty(api), return; end

pos = Position2Pixel(handles,api.getPosition());

sz = size(handles.Masks{nMask}.Mask);
switch handles.ViewDirection
  case 1, Mask = poly2mask(pos(:,1), pos(:,2), sz(2), sz(3));
  case 2, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(3));
  case 3, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(2));
end

handles.draw_mode = 'none';
handles = ApplySliceMask(handles, Mask, contour_mode);

ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function DrawFreehandContour(handles, contour_mode)
if isempty(handles.Masks), return; end
nMask = get(handles.lbMasks, 'Value');

handles.draw_mode = 'impoly';
guidata(handles.figure1, handles);

try
  h = imfreehand(handles.axes1);
catch
  return;
end

api = iptgetapi(h);
if isempty(api), return; end

pos = Position2Pixel(handles, api.getPosition());
sz = size(handles.Masks{nMask}.Mask);
switch handles.ViewDirection
  case 1, Mask = poly2mask(pos(:,1), pos(:,2), sz(2), sz(3));
  case 2, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(3));
  case 3, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(2));
end

handles.draw_mode = 'none';
handles = ApplySliceMask(handles, Mask, contour_mode);

ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)

for ii=1:length(handles.Masks)
  new_image = [];
  new_image.data = handles.Masks{ii}.Mask;
  new_image.ImageType = '3DMASK';
  new_image.Name = handles.Masks{ii}.Name;
  new_image.A = iff(handles.image.SlaveIdx > 0, handles.image.Aslave, eye(4));
  new_image.isStore = 1;
  arbuz_AddImage(handles.hh, new_image, handles.image.Image);
end

for ii=1:length(handles.XYZs)
  new_image = [];
  if handles.isSlave
    new_image.data = htransform_vectors(handles.image.Aslave, handles.XYZs{ii}.data);
  else
    new_image.data = handles.XYZs{ii}.data;
  end
  new_image.Name = handles.XYZs{ii}.Name;
  new_image.A = iff(handles.image.SlaveIdx > 0, handles.image.Aslave, eye(4));
  new_image.isStore = 1;
  new_image.ImageType = 'XYZ';
  arbuz_AddImage(handles.hh, new_image, handles.image.Image);
end

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbSaveSurface_Callback(hObject, eventdata, handles) %#ok<DEFNU>
if isempty(handles.image), return; end
nMask = get(handles.lbMasks, 'Value');

sz = handles.image.Box(1:3);
res = inputdlg({'Surface Name:', 'Max voxels in any dimension'}, 'Set Mask Name', 1, {handles.Masks{nMask}.Name, num2str(max(sz))});

if ~isempty(res{1})
  sz_factor = fix(sz ./ [str2double(res{2}), str2double(res{2}), str2double(res{2})]);
  sz_factor(sz_factor < 1) = 1;
  im_mask = handles.Masks{nMask}.Mask;
  Ascale = eye(4);

  if handles.is2D
    cont.Name = [res{1},'CTR'];
    im_mask = sum(im_mask, 3)  >0;

    if any(sz_factor ~= 1)
      new_sz   = fix(sz(1:2)./sz_factor(1:2));
      old_size = new_sz(1:2).*sz_factor(1:2);
      shape_array = [sz_factor(1:2); new_sz]; shape_array = shape_array(:)';
      compressed_mask = reshape(im_mask(1:old_size(1),1:old_size(2)), shape_array);
      compressed_mask = squeeze(sum(sum(sum(compressed_mask, 1), 3), 5));
      c = contourc(compressed_mask, [0.5, 0.5]);
      Ascale = hmatrix_translate(-[1 1 1]) * hmatrix_scale(sz_factor([2,1,3])) * hmatrix_translate([1 1 1]+(sz_factor-[1 1 1])/2);
    else
      c = contourc(double(im_mask), [0.5, 0.5]);
    end
    c(3,:) = 0;
    
    cont.data = c';
    cont.A = Ascale*iff(handles.image.SlaveIdx > 0, handles.image.Aslave, eye(4));
    cont.ImageType = 'CONTOUR';
    cont.isLoaded = 1;
    arbuz_AddImage(handles.hh, cont, handles.image.Image);
    
    
%     figure(100); clf; %imagesc(im_mask);
%     subplot(3,1,1); imagesc(im_mask); axis image
%     subplot(3,1,2); imagesc(compressed_mask); axis image
%     subplot(3,1,3);
%     arbuz_DrawContour2D(gca, struct('data', c'), eye(4));
%     f = contourc(double(im_mask), [0.5, 0.5]); f(3, :) = 0;
%     arbuz_DrawContour2D(gca, struct('data', f'), eye(4));
%     axis image
  else
    cont.Name = [res{1},'SRF'];
    
    if any(sz_factor ~= 1)
      new_sz   = fix(sz./sz_factor);
      old_size = new_sz.*sz_factor;
      shape_array = [sz_factor; new_sz]; shape_array = shape_array(:)';
      compressed_mask = reshape(im_mask(1:old_size(1),1:old_size(2),1:old_size(3)), shape_array);
      compressed_mask = squeeze(sum(sum(sum(compressed_mask, 1), 3), 5));
      [surface.face,surface.vert]=isosurface(compressed_mask, prod(sz_factor)/2);
      Ascale = hmatrix_translate(-[1 1 1]) * hmatrix_scale(sz_factor) * hmatrix_translate([1 1 1]+(sz_factor-[1 1 1])/2);
    else
      [surface.face,surface.vert]=isosurface(im_mask, 0.5);
    end
    
    cont.data = surface;
    cont.A = Ascale*iff(handles.image.SlaveIdx > 0, handles.image.Aslave, eye(4));
    cont.ImageType = '3DSURFACE';
    cont.isLoaded = 1;
    arbuz_AddImage(handles.hh, cont, handles.image.Image);
  end
  
  arbuz_UpdateInterface(handles.hh);
end

% --------------------------------------------------------------------
function pbCopy_Callback(hObject, eventdata, handles)
if isempty(handles.Masks), return; end
nMask = get(handles.lbMasks, 'Value');
handles.MaskBuffer = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN);
guidata(hObject, handles);

% --------------------------------------------------------------------
function pbPaste_Callback(hObject, eventdata, handles)
if isempty(handles.Masks) || isempty(handles.MaskBuffer), return; end
handles = ApplySliceMask(handles, handles.MaskBuffer, 'replace');
DrawSlice(handles);

% --------------------------------------------------------------------
function handles = ModeVisualizer(handles, new_mode)

% left_toolbar_buttons_size = size(handles.ToolBar);
% for ii=1:left_toolbar_buttons_size(1)
%   for jj=1:left_toolbar_buttons_size(2)
%     if ~isequal(handles.ToolBar{ii,jj}, -1) && isequal(handles.ToolBar.icon, new_mode)
%       
%     end
%   end
% end

% --------------------------------------------------------------------
function mDrawAdd_Callback(hObject, eventdata, handles)
handles = ModeVisualizer(handles, 'AddPoly');
DrawPolyContour(handles, 'add');

% --------------------------------------------------------------------
function mDrawAddFreehand_Callback(hObject, eventdata, handles)
handles = ModeVisualizer(handles, 'AddFreehand');
DrawFreehandContour(handles, 'add');

% --------------------------------------------------------------------
function mDrawErase_Callback(hObject, eventdata, handles)
handles = ModeVisualizer(handles, 'RemovePoly');
DrawPolyContour(handles, 'erase');

% --------------------------------------------------------------------
function mDrawEraseFreehand_Callback(hObject, eventdata, handles)
handles = ModeVisualizer(handles, 'EraseFreehand');
DrawFreehandContour(handles, 'erase');

% --------------------------------------------------------------------
function mDrawPencil_Callback(hObject, eventdata, handles)
if isempty(handles.Masks), return; end
nMask = get(handles.lbMasks, 'Value');

[x, y] = ginput_workaround(handles, 1);
[x, y] = Position2Pixel(handles,[x, y]);

sz = size(handles.Masks{nMask}.Mask);

switch handles.ViewDirection
  case 1, Mask = false([sz(2), sz(3)]); Mask(round(y), round(x))=true;
  case 2, Mask = false([sz(1), sz(3)]); Mask(round(y), round(x))=true;
  case 3, Mask = false([sz(1), sz(2)]); Mask(round(y), round(x))=true;
end

handles = ApplySliceMask(handles, Mask, 'add');
DrawSlice(handles);

% --------------------------------------------------------------------
function rbApplyAll_Callback(hObject, eventdata, handles)
hh = [handles.rbApplyAll, handles.rbApplyAllNorm, handles.rbApplySlice];
set(hh, 'Value', 0);
set(hObject, 'Value', 1)
handles.ApplyTo = find(hh == hObject);
guidata(hObject, handles);

% --------------------------------------------------------------------
function rbViewDirection_Callback(hObject, eventdata, handles)
hh = [handles.rbView1, handles.rbView2, handles.rbView3];
set(hh, 'Value', 0);
set(hObject, 'Value', 1)
handles.ViewDirection = find(hh == hObject);
handles.SliceN = SetSlice(handles, 0);
nSlices = GetSliceNumber(handles);
set(handles.slSlice, 'Max', nSlices, 'Min', 1, 'SliderStep', min(1/(nSlices-1)*[1,10],[1,1]));

% reset zoom parameters
[zoom, handles.box_min, handles.box_max] = GetImageDims(handles);
handles.idx1 = true(zoom(1), 1);
handles.idx2 = true(zoom(2), 1);
handles.ax1 = linspace(handles.box_min(1), handles.box_max(1), ...
  zoom(1));
handles.ax2 = linspace(handles.box_min(2), handles.box_max(2), ...
  zoom(2));

guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----- S E R V I C E     F U N C T I O N S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function find_list = GetSelectedImage(handles)
find_list = arbuz_FindImage(handles.hh, 'all', 'Highlighted', 1, {});

if isempty(find_list)
  disp('Image2ContourPLG: No 3D image data found.');
else
  find_list = find_list{1};
end

% --------------------------------------------------------------------
function SliceN = SetSlice(handles, n_inc, n_set)

try
  val(1) = str2double(get(handles.eSliceN1, 'String'));
  val(2) = str2double(get(handles.eSliceN2, 'String'));
  val(3) = str2double(get(handles.eSliceN3, 'String'));
catch err
  val = [1,1,1];
end

N = GetSelectedAxis(handles);
if ~isempty(N)
  if ~isempty(n_inc)
    val(N) = val(N) + n_inc;
  else
    val = n_set;
  end
end

if isfield(handles, 'ax')
  for ii=1:3
    MaxN = handles.ax{ii}.n;
    if val(ii) >= 1 && val(ii) <= MaxN, SliceN(ii) = val(ii);
    elseif val(ii) < 1, SliceN(ii) = MaxN;
    else
      SliceN(ii) = 1;
    end
  end
else
  SliceN = val;
end

set(handles.eSliceN1, 'String', num2str(SliceN(1)));
set(handles.eSliceN2, 'String', num2str(SliceN(2)));
set(handles.eSliceN3, 'String', num2str(SliceN(3)));
  
% --------------------------------------------------------------------
function n = GetSliceNumber(handles)
n = size(handles.image.data, handles.ViewDirection);

% --------------------------------------------------------------------
function [AXX] = GetImageDims(handles)
AXX = cell(3,1);
n = size(handles.image.data);
for ii=1:3, AXX{ii}.n = n(ii); end

box_max = [handles.image.Box(1:3), 1]*handles.image.Anative; 
box_min = [1,1,1,1]*handles.image.Anative;
for ii=1:3, AXX{ii}.min = box_min(ii); AXX{ii}.max = box_max(ii); end

for ii=1:3
  AXX{ii}.zoom = AXX{ii}.n;
  AXX{ii}.xx = linspace(AXX{ii}.min, AXX{ii}.max, AXX{ii}.n); 
  AXX{ii}.idx = true(AXX{ii}.zoom, 1);
end

AXX{1}.axis = [2,3];
AXX{2}.axis = [1,3];
AXX{3}.axis = [1,2];

AXX{1}.axis_names = ['Y','Z'];
AXX{2}.axis_names = ['X','Z'];
AXX{3}.axis_names = ['X','Y'];

AXX{1}.axis_pos = 1;
AXX{2}.axis_pos = 2;
AXX{3}.axis_pos = 3;

AXX{1}.axes = handles.axes1;
AXX{2}.axes = handles.axes2;
AXX{3}.axes = handles.axes3;

% --------------------------------------------------------------------
function [handles, mm] = GetMask(handles, mask_name, create_new_mask)

for ii=1:length(handles.Masks)
  if strcmp(handles.Masks{ii}.Name, mask_name)
    mm = ii; return;
  end
end

if create_new_mask
  handles.Masks{end+1} = struct('Name', mask_name, 'Mask', false(size(handles.image.data)));
  guidata(handles.figure1, handles);
  ListAllMasks(handles);
  mm = length(handles.Masks);
else
  mm = [];
end

% --------------------------------------------------------------------
function ListAllMasks(handles)
str = {};
for ii=1:length(handles.Masks)
  the_name  = safeget(handles.Masks{ii}, 'Name', '??');
  [the_color] = arbuz_Color('', ii);
  color_structure = safeget(handles.Masks{ii}, 'Color', []); 
  the_color = safeget(color_structure, 'Color', the_color);
  if strcmp(the_color,  'blue'), the_color = [0, 0, 1]; end
  if strcmp(the_color,  'red'), the_color = [1, 0, 0]; end
  if strcmp(the_color,  'green'), the_color = [0, 1, 0]; end
  the_html_color = sprintf('%02x%02x%02x', fix(the_color(1)*255), fix(the_color(2)*255), fix(the_color(3)*255));
  str{end+1} = ['<html><font color=',the_html_color,'>',the_name,'</font></html>'];
end
if isempty(str), str = {'No masks'}; end
val = get(handles.lbMasks, 'Value');
set(handles.lbMasks, 'String', str, 'Value', min([val, length(str)]));

str = {};
for ii=1:length(handles.XYZs)
  str{end+1} = safeget(handles.XYZs{ii}, 'Name', '??');
end

% --------------------------------------------------------------------
function DrawSlice(handles)
cla(handles.axes1);
cla(handles.axes2);
cla(handles.axes3);

opt = safeget(handles, 'opt', []);
vis_abs = safeget(opt, 'vis_abs', 0);
vis_low = safeget(opt, 'vis_low', 0);
vis_high = safeget(opt, 'vis_high', 1);

for ii=1:3
  hh = handles.ax{ii}.axes;
  aa = handles.ax{ii}.axis;
  idx1 = handles.ax{aa(1)}.idx;
  idx2 = handles.ax{aa(2)}.idx;
  ax1 = handles.ax{aa(1)}.xx(idx1);
  ax2 = handles.ax{aa(2)}.xx(idx2);
  switch ii
    case 1
      imagesc(ax1, ax2, squeeze(handles.image.data(handles.SliceN(1),idx1, idx2)), 'Parent', hh);
    case 2
      imagesc(ax1, ax2, squeeze(handles.image.data(idx1, handles.SliceN(2), idx2)), 'Parent', hh);
    case 3
      imagesc(ax1, ax2, squeeze(handles.image.data(idx1, idx2, handles.SliceN(3))), 'Parent', hh);
  end
  axis(hh, 'equal'); axis(hh, 'image');
  
  if vis_abs
    set(hh,'CLim',[vis_low, vis_high], 'YDir', 'normal', 'XDir', 'normal');
  else
    dclim = cast(handles.image.max - handles.image.min, 'double');
    set(hh,'CLim',handles.image.min + dclim*[vis_low, vis_high], 'YDir', 'normal', 'XDir', 'normal');
  end
  
  hold(hh,'on');
  switch safeget(opt, 'colormap', 2)
    case 1, colormap jet; MaskColors = {'w', 'k', 'y', 'm', 'w', 'k', 'y', 'm', 'w', 'k', 'y', 'm', 'w', 'k', 'y', 'm'};
    case 2, colormap bone; MaskColors = {'r', 'b', 'm', 'y', 'r', 'b', 'y', 'm', 'r', 'b', 'm', 'y', 'r', 'b', 'y', 'm'};
    case 3, colormap hot; MaskColors = {'r', 'b', 'y', 'm', 'r', 'b', 'y', 'm', 'r', 'b', 'm', 'y', 'r', 'b', 'y', 'm'};
    case 4, colormap copper; MaskColors = {'r', 'b', 'y', 'm', 'r', 'b', 'y', 'm', 'r', 'b', 'm', 'y', 'r', 'b', 'y', 'm'};
    case 5, colormap pink; MaskColors = {'r', 'b', 'y', 'm', 'r', 'b', 'y', 'm', 'r', 'b', 'm', 'y', 'r', 'b', 'y', 'm'};
  end
  
  nMask = get(handles.lbMasks, 'Value');
  
  legend_str = {};
  for jj=1:length(handles.Masks)
    [Mask] = GetMask2D(handles.Masks{jj}.Mask, ii, handles.SliceN(ii));
    
    if any(Mask(:))
      legend_str{end+1} = safeget(handles.Masks{jj}, 'Name', '?');
      the_color = arbuz_Color('', jj);
      color_structure = safeget(handles.Masks{jj}, 'Color', []);
      
      [~, h] = contour(ax2, ax1, Mask(idx1, idx2), [0.5 0.5], 'Parent', hh);
      set(h, 'LineColor', safeget(color_structure, 'Color', the_color));
      if jj==nMask
        set(h, 'Linewidth', 2.5);
      else
        set(h, 'Linewidth', 1);
      end
    end
  end
  
  
  csize  = abs(max(ax1)-min(ax1)) / 60;
  csizey = csize;
  if handles.is2D, csizey = -csize; end
  
%   if ~isempty(legend_str), legend(legend_str,'Location', 'NorthEast', 'Parent', hh); end
  
  set(hh, 'ydir', 'reverse');
end

% --------------------------------------------------------------------
function Mask2D = GetMask2D(Mask3D, view_direction, sliceN)
if isempty(Mask3D), Mask2D = []; return; end

switch view_direction
  case 1, Mask2D = squeeze(Mask3D(sliceN, :, :));
  case 2, Mask2D = squeeze(Mask3D(:, sliceN, :));
  case 3, Mask2D = Mask3D(:, :, sliceN);
end

% --------------------------------------------------------------------
function Mask3D = SetMask2D(Mask3D, view_direction, Mask2D, sliceN)
switch view_direction
  case 1, Mask3D(sliceN, :, :) = Mask2D;
  case 2, Mask3D(:, sliceN, :) = Mask2D;
  case 3, Mask3D(:, :, sliceN) = Mask2D;
end

% --------------------------------------------------------------------
function image_2D = GetImage2D(handles, SliceN)
switch handles.ViewDirection
  case 1, image_2D = squeeze(handles.image.data(SliceN,:,:));
  case 2, image_2D = squeeze(handles.image.data(:,SliceN, :));
  case 3, image_2D = handles.image.data(:,:, SliceN); 
end

% --------------------------------------------------------------------
function data = GetImage3D(handles, data)
switch handles.ViewDirection
  case 1, data = permute(data, [2,3,1]);
  case 2, data = permute(data, [1,3,2]);
end

% --------------------------------------------------------------------
function data = GetImage4D(handles, data)
switch handles.ViewDirection
  case 1, data = permute(data, [2,3,1,4]);
  case 2, data = permute(data, [1,3,2,4]);
end

% --------------------------------------------------------------------
function im_mask = SetMask3D(handles, im_mask)
switch handles.ViewDirection
  case 1, im_mask = permute(im_mask, [3,1,2]);
  case 2, im_mask = permute(im_mask, [1,3,2]);
end

% --------------------------------------------------------------------
function point = pos2Dto3D(handles,x,y)

switch handles.ViewDirection
  case 1, point = [handles.SliceN, x, y];
  case 2, point = [y, handles.SliceN, x];
  case 3, point = [y, x, handles.SliceN];
end

% --------------------------------------------------------------------
function [val, point] = pos2DtoVal(handles,x,y)

point = pos2Dto3D(handles,x,y);
val = [];
if any(point < 1) || any(point > handles.image.Box(1:3)), return; end

val = handles.image.data(point(1), point(2), point(3));

% --------------------------------------------------------------------
function handles = ApplySliceMask(handles, Mask, contour_mode)
if isempty(handles.Masks), return; end
nMask = get(handles.lbMasks, 'Value');
sz1 = size(handles.Masks{nMask}.Mask);

switch handles.ViewDirection
  case 1, Mask = reshape(Mask, [1, sz1(2), sz1(3)]);
  case 2, Mask = reshape(Mask, [sz1(1), 1, sz1(3)]);
  case 3, Mask = reshape(Mask, [sz1(1), sz1(2), 1]);
end

switch contour_mode
  case 'replace',
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = Mask;
      case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = Mask;
      case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = Mask;
    end
  case 'add',
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = ...
          handles.Masks{nMask}.Mask(handles.SliceN, :,:) | Mask;
      case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = ...
          handles.Masks{nMask}.Mask(:, handles.SliceN,:) | Mask;
      case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = ...
          handles.Masks{nMask}.Mask(:,:, handles.SliceN) | Mask;
    end
  case 'erase',
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = ...
          handles.Masks{nMask}.Mask(handles.SliceN, :,:) & (~Mask);
      case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = ...
          handles.Masks{nMask}.Mask(:, handles.SliceN,:) & (~Mask);
      case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = ...
          handles.Masks{nMask}.Mask(:,:, handles.SliceN) & (~Mask);
    end
end

StoreMasks(handles);

% --------------------------------------------------------------------
function KeyPressFunction(src,evnt)
handles = guidata(src);

k= evnt.Key; %k is the key that is pressed

if strcmp(k,'leftarrow'), pbPrevSlice_Callback(handles.pbPrevSlice, [], handles);
elseif strcmp(k,'rightarrow'), pbNextSlice_Callback(handles.pbNextSlice, [], handles);
end

% --------------------------------------------------------------------
function mSpecialOutline0p15_Callback(hObject, eventdata, handles)
[handles,idx] = GetMask(handles, 'Outline', true);

data = GetImage3D(handles, handles.image.data);

res=inputdlg({'Threshold'}, 'Enter parameters',1, {'0.15'});
threshold = str2num(res{1});

switch handles.ApplyTo
  case 1, im_mask = outside_mask3(double(data), threshold);
  case 2, im_mask = outside_mask3(double(data), threshold);
    for ii=1:size(data,3)
      im_mask(:,:,ii) = outside_mask3(data(:,:,ii), threshold);
    end
  case 3,
    im_mask = GetImage3D(handles, handles.Masks{idx}.Mask);
    im_mask(:,:,handles.SliceN) = outside_mask3(data(:,:,handles.SliceN), str2num(res{1}));
end

handles.Masks{idx}.Mask = SetMask3D(handles, im_mask);
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mSpecialSelectAll_Callback(hObject, eventdata, handles)

res=inputdlg({'Low threshold:', 'High threshold:', 'Morphology open/close/clean:', 'Name:'}, ...
  'Enter parameters',1, {'0.15', '1', 'clean', 'All'});
if isempty(res), return; end

low_threshold = str2num(res{1});
high_threshold = str2num(res{2});
morph_protocol = res{3};
morph_open = ~isempty(strfind(morph_protocol, 'open'));
morph_close = ~isempty(strfind(morph_protocol, 'close'));
morph_clean = ~isempty(strfind(morph_protocol, 'clean'));
[handles,idx] = GetMask(handles, res{4}, true);

data = GetImage3D(handles, handles.image.data);

switch handles.ApplyTo
  case 1
    im_mask = data >= low_threshold*handles.image.max & data <= high_threshold*handles.image.max;
    for ii=1:size(data,3)
      if morph_open, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open'); end
      if morph_close, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close'); end;
      if morph_clean, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'clean'); end
    end
  case 2
    im_mask = false(size(data));
    for ii=1:size(data,3)
      mmax = max(max(data(:,:,ii)));
      im_mask(:,:,ii) = data(:,:,ii) >= low_threshold*mmax & data(:,:,ii) <= high_threshold*mmax;
      if morph_open, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open'); end
      if morph_close, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close'); end
      if morph_clean, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'clean'); end
    end
  case 3
    im_mask = GetImage3D(handles, handles.Masks{idx}.Mask);
    mmax = max(max(data(:,:,handles.SliceN)));
    im_mask(:,:,handles.SliceN) = data(:,:,handles.SliceN) >= low_threshold*mmax & data(:,:,handles.SliceN) <= high_threshold*mmax;
    if morph_open, im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'open'); end
    if morph_close, im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'close'); end
      if morph_clean, im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'clean'); end
end

handles.Masks{idx}.Mask = SetMask3D(handles, im_mask);
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mSpecialFiducials_Callback(hObject, eventdata, handles)
[handles,idx] = GetMask(handles, 'FID', true);

data = GetImage3D(handles, handles.image.data);

res=inputdlg({'Threshold:', 'Exclude mask:'}, 'Enter parameters',1, {'0.15','Outline'});
if isempty(res), return; end

threshold = str2double(res{1});
exclude_mask_name = res{2};

[handles,idx1] = GetMask(handles, exclude_mask_name, false);
if isempty(idx1)
  exclude_mask = outside_mask3(data, threshold);
else
  exclude_mask = GetImage3D(handles, handles.Masks{idx1}.Mask);
end

im_mask = data > threshold*max(data(:));
im_mask(exclude_mask) = false;

for ii=1:size(data,3)
  im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open');
  im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close');
end

im_mask(exclude_mask) = false;

im_mask = SetMask3D(handles, im_mask);

handles.Masks{idx}.Mask = im_mask;
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function args = GetArgs(handles)

args = [];

mask_str ={};
for ii=1:length(handles.Masks);mask_str{end+1}=handles.Masks{ii}.Name; end
[s2,v] = listdlg('PromptString','Select argument 2:','SelectionMode','single',...
  'ListString',mask_str);
if ~v, return; end

args = [s2];

% --------------------------------------------------------------------
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)

[n, C] = GetSelectedAxis(handles);
axes = [handles.axes1, handles.axes2, handles.axes3];
NN = 1:3;
if ~isempty(n)
  axis_names = handles.ax{n}.axis_names;
  set(axes(n),'box','on');
  set(axes(NN(NN~=n)),'box','off');
  txt = sprintf('%c:%5.2f, %c:%5.2f', axis_names(1), C(1,1), axis_names(2), C(2,2));
  set(handles.txtMousePosition, 'String', txt);
%   CoordXY = C(1,1:2);
%   CXY = floor(Position2Pixel(handles,ax,CoordXY)+0.5);
else
  set(axes,'box','off');
  set(handles.txtMousePosition, 'String', '');
end

% --------------------------------------------------------------------
function figure1_WindowButtonUpFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
[n, C] = GetSelectedAxis(handles);
axes = [handles.axes1, handles.axes2, handles.axes3];
if ~isempty(n)
  handles.SliceN = SetSlice(handles, [], floor(Position2Pixel(handles,n,C(1,1:2))+0.5));
  DrawSlice(handles);
else
  set(axes,'box','off');
  set(handles.txtMousePosition, 'String', '');
end

% --------------------------------------------------------------------
function [N, CC] = GetSelectedAxis(handles)
N = [];
CC = [];
axes = {handles.axes1, handles.axes2, handles.axes3};

for ii=1:3
  C = get(axes{ii},'currentpoint');
  xlim = get(axes{ii},'xlim');
  ylim = get(axes{ii},'ylim');
  outX = ~any(diff([xlim(1) C(1,1) xlim(2)])<0);
  outY = ~any(diff([ylim(1) C(1,2) ylim(2)])<0);
  if outX&outY
    N = ii;
    CC = C;
  end
end

% --------------------------------------------------------------------
function cursors = DrawCursor(hh, x, y, r, cursors)
if ~isempty(cursors), delete(cursors(ishandle(cursors))); end
cursors = [];
a = [0:.1:2*pi, 0];
cursors(end+1) = plot(x+r*cos(a), y+r*sin(a), 'g', 'Parent', hh, 'Tag', 'CursorElements');

% --------------------------------------------------------------------
function mDrawRubber_Callback(hObject, eventdata, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);

handles.op_mode = 'rubber';

pos = get(handles.pCustomTools, 'Position');
set(handles.pCustomTools, 'Title', 'Rubber radius');
handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0.2, 'Max', 5, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', handles.rubber_radius, ...
  'Callback', 'TTouchImageBasedPLG(''ff_rubber_size'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mDrawBrush_Callback(hObject, eventdata, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'brush';

pos = get(handles.pCustomTools, 'Position');
set(handles.pCustomTools, 'Title', 'Brush radius');
handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', .2, 'Max', 5, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', handles.rubber_radius, ...
  'Callback', 'TTouchImageBasedPLG(''ff_rubber_size'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mSpecialVisrange_Callback(hObject, eventdata, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'vis_range';

pos = get(handles.pCustomTools, 'Position');
opt = safeget(handles, 'opt', []);
set(handles.pCustomTools, 'Title', 'Visualization ranges');

  if safeget(opt, 'vis_abs', 0)
    setmin = handles.image.min;
    setmax = handles.image.max;
  else
    setmin = 0;
    setmax = 1;
  end
  
handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', setmin, 'Max', setmax, ...
  'Units', 'pixels', 'Position', [10, 30, pos(3)-20, 15], 'Value', safeget(opt, 'vis_low', 0), ...
  'Callback', 'TTouchImageBasedPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', setmin, 'Max', setmax, ...
  'Units', 'pixels', 'Position', [10, 50, pos(3)-20, 15], 'Value', safeget(opt, 'vis_high', 1), ...
  'Callback', 'TTouchImageBasedPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{3} = uicontrol('Style','checkbox', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 10, 60, 15], 'Value', safeget(opt, 'vis_abs', 0), ...
  'Callback', 'TTouchImageBasedPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{4} = uicontrol('Style','edit', 'Parent', handles.pCustomTools,  ...
  'Units', 'pixels', 'Position', [30, 10, 55, 15], 'string', num2str(safeget(opt, 'vis_low', 0)), ...
  'Callback', 'TTouchImageBasedPLG(''ff_vis_range'',gcbo,guidata(gcbo))');
handles.op_control{5} = uicontrol('Style','edit', 'Parent', handles.pCustomTools,  ...
  'Units', 'pixels', 'Position', [95, 10, 55, 15], 'string', num2str(safeget(opt, 'vis_low', 0)), ...
  'Callback', 'TTouchImageBasedPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);
ff_vis_range(handles.op_control{1}, handles);

% --------------------------------------------------------------------
function mVisualColormap_Callback(hObject, eventdata, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'vis_cmap';

pos = get(handles.pCustomTools, 'Position');
opt = safeget(handles, 'opt', []);
set(handles.pCustomTools, 'Title', 'Select colormap');
handles.op_control{1} = uicontrol('Style','popupmenu', 'Parent', handles.pCustomTools, ...
  'String', {'jet', 'bone', 'hot', 'copper', 'pink'}, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', safeget(opt, 'colormap', 2), ...
  'Callback', 'TTouchImageBasedPLG(''ff_cmap'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mSpecialInteractiveThreshold_Callback(hObject, eventdata, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'intrv_threshold';

pos = get(handles.pCustomTools, 'Position');
opt = safeget(handles, 'opt', []);
set(handles.pCustomTools, 'Title', 'Interactive thresholds');
handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', safeget(opt, 'low_threshold', 0), ...
  'Callback', 'TTouchImageBasedPLG(''ff_intrv_threshold'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 40, pos(3)-20, 20], 'Value', safeget(opt, 'high_threshold', 1), ...
  'Callback', 'TTouchImageBasedPLG(''ff_intrv_threshold'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function handles=ClearCustomTools(handles)
op_control = safeget(handles, 'op_control', []);
for ii=1:length(op_control)
  if ishandle(op_control{ii}), delete(op_control{ii}); end
end
handles.op_control = [];
handles.op_mode = 'none';
set(handles.pCustomTools, 'Title', 'Custom tools');
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function ff_rubber_size(hObject, handles)
handles.rubber_radius = get(hObject, 'Value');
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function ff_strel_size(hObject, handles)
handles.strel_disk_radius = get(hObject, 'Value');
guidata(handles.figure1, handles);
disp(['Radius: ', num2str(handles.strel_disk_radius)])

% --------------------------------------------------------------------
function ff_flood_threshold(hObject, handles)
handles.flood_threshold = get(hObject, 'Value');
guidata(handles.figure1, handles);
disp(['Threshold: ', num2str(handles.flood_threshold)])

% --------------------------------------------------------------------
function ff_vis_range(hObject, handles)
handles.opt.vis_abs = get(handles.op_control{3}, 'Value');

if strcmp(get(hObject, 'Style'), 'checkbox')
  
  if handles.opt.vis_abs
    set(handles.op_control{1}, 'Min', handles.image.min, 'Max', handles.image.max, 'Value' , handles.image.min);
    set(handles.op_control{2}, 'Min', handles.image.min, 'Max', handles.image.max, 'Value' , handles.image.max);
  else
    set(handles.op_control{1}, 'Min', 0, 'Max', 1, 'Value' , 0);
    set(handles.op_control{2}, 'Min', 0, 'Max', 1, 'Value' , 1);
  end
  handles.opt.vis_low = get(handles.op_control{1}, 'Value');
  handles.opt.vis_high = get(handles.op_control{2}, 'Value');
  set(handles.op_control{4}, 'string', num2str(handles.opt.vis_low));
  set(handles.op_control{5}, 'string', num2str(handles.opt.vis_high));
  
elseif strcmp(get(hObject, 'Style'), 'slider')
  
  handles.opt.vis_low = get(handles.op_control{1}, 'Value');
  handles.opt.vis_high = get(handles.op_control{2}, 'Value');
  
  set(handles.op_control{4}, 'string', num2str(handles.opt.vis_low));
  set(handles.op_control{5}, 'string', num2str(handles.opt.vis_high));

elseif strcmp(get(hObject, 'Style'), 'edit')
  
  handles.opt.vis_low = str2double(get(handles.op_control{4}, 'string'));
  handles.opt.vis_high = str2double(get(handles.op_control{5}, 'string'));
  
  handles.opt.vis_low = max([handles.opt.vis_low,handles.image.min]);
  set(handles.op_control{4}, 'string', num2str(handles.opt.vis_low));

  handles.opt.vis_high = min([handles.opt.vis_high,handles.image.max]);
  set(handles.op_control{5}, 'string', num2str(handles.opt.vis_high));
  
  set(handles.op_control{1}, 'Value', handles.opt.vis_low);
  set(handles.op_control{2}, 'Value', handles.opt.vis_high); 
end

guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function ff_cmap(hObject, handles)
handles.opt.colormap = get(hObject, 'Value');
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function ff_intrv_threshold(hObject, handles)
handles.opt.low_threshold = get(handles.op_control{1}, 'Value');
handles.opt.high_threshold = get(handles.op_control{2}, 'Value');
nMask = get(handles.lbMasks, 'Value');

data = GetImage3D(handles, handles.image.data);

switch handles.ApplyTo
  case 1
    im_mask = data >= handles.opt.low_threshold*handles.image.max & data <= handles.opt.high_threshold*handles.image.max;
    for ii=1:size(data,3)
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open');
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close');
    end
  case 2
    im_mask = false(size(data));
    for ii=1:size(data,3)
      mmax = max(max(data(:,:,ii)));
      im_mask(:,:,ii) = data(:,:,ii) >= handles.opt.low_threshold*mmax & data(:,:,ii) <= handles.opt.high_threshold*mmax;
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open');
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close');
    end
  case 3
    im_mask = GetImage3D(handles, handles.Masks{nMask}.Mask);
    mmax = max(max(data(:,:,handles.SliceN)));
    im_mask(:,:,handles.SliceN) = data(:,:,handles.SliceN) >= handles.opt.low_threshold*mmax & data(:,:,handles.SliceN) <= handles.opt.high_threshold*mmax;
    im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'open');
    im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'close');
end

handles.Masks{nMask}.Mask = SetMask3D(handles, im_mask);
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mSpecialRemoveOutsideOutline_Callback(hObject, eventdata, handles)

nMask = get(handles.lbMasks, 'Value');

[handles,idx] = GetMask(handles, 'Outline', false);
if isempty(idx)
  im_mask = GetImage3D(handles, handles.Masks{nMask}.Mask);
  im_mask = im_mask & outside_mask3(GetImage3D(handles, handles.image.data), 0.15);
  handles.Masks{nMask}.Mask = SetMask3D(handles, im_mask);
else
  handles.Masks{nMask}.Mask = handles.Masks{nMask}.Mask & GetImage3D(handles, handles.Masks{idx}.Mask);
end

guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, eventdata.VerticalScrollCount);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mDrawDelete_Callback(hObject, eventdata, handles)
nMask = get(handles.lbMasks, 'Value');
im_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN);

[x, y] = ginput_workaround(handles, 1);
[x, y] = Position2Pixel(handles,[x, y]);
objects = bwlabel(im_mask);
remove_mask_idx = objects(fix(y), fix(x));
im_mask(objects == remove_mask_idx) = false;

handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, im_mask, handles.SliceN);
StoreMasks(handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mDrawDeleteAll_Callback(hObject, eventdata, handles)
nMask = get(handles.lbMasks, 'Value');

if handles.ApplyTo == 3
  im_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN);
  im_mask = false(size(im_mask));
  handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, im_mask, handles.SliceN);
else
  im_mask = GetImage3D(handles, handles.Masks{nMask}.Mask);
  im_mask = false(size(im_mask));
  handles.Masks{nMask}.Mask = SetMask3D(handles, im_mask);
end
StoreMasks(handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mMath_Callback(hObject, eventdata, handles)
nMask = get(handles.lbMasks, 'Value');
arg1 = handles.Masks{nMask}.Mask;

switch hObject
  % single argument opearations
  case handles.mMathInflate
    se = epr_strel('sphere', 1);
    handles.Masks{nMask}.Mask = imdilate(arg1, se);
    StoreMasks(handles);
    DrawSlice(handles);
  case handles.mMathDeflate
    se = epr_strel('sphere', 1);
    handles.Masks{nMask}.Mask = imerode(arg1, se);
    StoreMasks(handles);
    DrawSlice(handles);

  % two argument opearations
  otherwise
    args = GetArgs(handles);
    
    if ~isempty(args)
      arg2 = handles.Masks{args}.Mask;
      switch hObject
        case handles.mMathAND
          res = and(arg1, arg2);
        case handles.mMathOR
          res = or(arg1, arg2);
        case handles.mMathRemoveOverlap
          res = arg1 & (~arg2);
      end
      switch handles.ApplyTo
        case {1,2}
          handles.Masks{nMask}.Mask = res;
        case 3
          im_mask = GetImage3D(handles, handles.Masks{nMask}.Mask);
          im_new_mask = GetImage3D(handles, res);
          im_mask(:,:,handles.SliceN) = im_new_mask(:,:,handles.SliceN);
          handles.Masks{nMask}.Mask = SetMask3D(handles, im_mask);
      end
    end
    
    StoreMasks(handles);
    DrawSlice(handles);
end

% --------------------------------------------------------------------
function mSpecialFloodFill_Callback(hObject, eventdata, handles)
if length(handles.Masks) < 1
  disp('No mask is selected.');
  return;
end

flood_threshold = safeget(handles, 'flood_threshold', 3);
if ~strcmp(handles.op_mode, 'ff_flood_threshold')
  handles=ClearCustomTools(handles);
  handles.op_mode = 'ff_flood_threshold';
  
  pos = get(handles.pCustomTools, 'Position');
  set(handles.pCustomTools, 'Title', 'Flood threshold [std]');
  handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 4, ...
    'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', flood_threshold, ...
    'Callback', 'TTouchImageBasedPLG(''ff_flood_threshold'',gcbo,guidata(gcbo))');
end

[x, y] = ginput_workaround(handles, 1);
[x, y] = Position2Pixel(handles,[x, y]);

nMask = get(handles.lbMasks, 'Value');

original_image = GetImage2D(handles, handles.SliceN);
current_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN);
seed_mask = false(size(original_image));
seed_mask(fix(y), fix(x)) = true;
if any(seed_mask(:)&current_mask(:)), seed_mask = current_mask; end
out_mask = epr_AutoMask2D('FloodFillAdjusted', original_image, seed_mask, flood_threshold);
out_mask = epr_AutoMask2D('Largest', [], out_mask, []);
out_mask = out_mask | current_mask;
handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, out_mask, handles.SliceN);

%   idx = pos2Dto3D(handles, fix(x),fix(y));
%   subset = handles.image.data(idx(1)+(-2:2), idx(2)+(-2:2), idx(3)+(-2:2));
%   mean_subset = mean(subset(:)); std_subset = std(subset(:));
%   res = inputdlg({'Low threshold', 'High threshold', 'Mask'}, ...
%     'Settings', 1, {num2str(mean_subset - 2*std_subset),num2str(mean_subset + 2*std_subset), handles.Masks{nMask}.Name});
%   
%   if ~isempty(res)
%     low_threshold = str2double(res{1});
%     high_threshold = str2double(res{2});
%     
%     mask = zeros(size(handles.image.data));
%     mask(handles.image.data > low_threshold & handles.image.data < high_threshold) = 1;
%     
%     Extract = FloodFill3D(mask, idx);
%     [handles,idx1] = GetMask(handles, res{3}, true);
%     handles.Masks{nMask}.Mask = handles.Masks{idx1}.Mask | Extract;
%   end

StoreMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function slSlice_Callback(hObject, ~, handles)
val = fix(get(handles.slSlice, 'Value'));

N = GetSliceNumber(handles);
if val >= 1 && val <= N, SliceN = val;
elseif val < 1, SliceN = N;
else
  SliceN = 1;
end

handles.SliceN = SliceN;
set(handles.eSliceN1, 'String', num2str(SliceN));
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mVisualZoom_Callback(hObject, ~, handles)
xy = getrect;
handles.idx2 = handles.ax2 >= xy(1) & handles.ax2 <= xy(1)+xy(3);
handles.idx1 = handles.ax1 >= xy(2) & handles.ax1 <= xy(2)+xy(4);

guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mVisualZoomOut_Callback(hObject, ~, handles)
handles.idx1 = true(size(handles.ax1));
handles.idx2 = true(size(handles.ax2));

guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mAuto_Callback(hObject, eventdata, handles)
if length(handles.Masks) < 1
  disp('No mask is selected.');
  return;
end

nMask = get(handles.lbMasks, 'Value');

handles.strel_disk_radius = fix(safeget(handles, 'strel_disk_radius', 1));
if ~strcmp(handles.op_mode, 'ff_strel_size')
  handles=ClearCustomTools(handles);
  handles.op_mode = 'ff_strel_size';
  
  pos = get(handles.pCustomTools, 'Position');
  set(handles.pCustomTools, 'Title', 'Active radius [voxel]');
  handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 1, 'Max', 8, 'SliderStep', 1/7*[1,1], ...
    'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', handles.strel_disk_radius, ...
    'Callback', 'TTouchImageBasedPLG(''ff_strel_size'',gcbo,guidata(gcbo))');
end

if ~isempty(eventdata)
  switch eventdata
    case 2, hObject = handles.mAutoClose;
    case 1, hObject = handles.mAutoOpen;
  end
end

se = strel('disk', handles.strel_disk_radius);
if handles.ApplyTo == 3
  current_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN);
  switch hObject
    case handles.mAutoOpen, current_mask = imopen(current_mask, se);
    case handles.mAutoClose, current_mask = imclose(current_mask, se);
    case handles.mAutoLargest, current_mask = epr_AutoMask2D('Largest', [], current_mask, []);
  end
  handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, current_mask, handles.SliceN);
else
  for ii=1:GetSliceNumber(handles)
    current_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, ii);
    switch hObject
      case handles.mAutoOpen, current_mask = imopen(current_mask, se);
      case handles.mAutoClose, current_mask = imclose(current_mask, se);
      case handles.mAutoLargest, current_mask = epr_AutoMask2D('Largest', [], current_mask, []);
        case handles.mAutoFillHoles, current_mask = imfill(current_mask, 'holes');
    end
    handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, current_mask, ii);
  end
end
StoreMasks(handles);
DrawSlice(handles);

disp('Operation finished.');

% --------------------------------------------------------------------
function mAutoPropogate_Callback(hObject, ~, handles)
if length(handles.Masks) < 1
  disp('No mask is selected.');
  return;
end

nMask = get(handles.lbMasks, 'Value');
se = strel('disk', 1);

% split current_mask into objects
[labeled_masks, n_labeled_masks] = bwlabel(GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN));

for jj=1:n_labeled_masks
  current_mask = labeled_masks == jj;
  n_el = numel(find(current_mask));
  n_slices_propogated = 0;
  for ii=handles.SliceN+1:GetSliceNumber(handles)
    original_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, ii);
    original_image = GetImage2D(handles, ii);
    current_mask = epr_AutoMask2D('FloodFillAdjusted', original_image, current_mask, safeget(handles, 'flood_threshold', 3));
    current_mask = epr_AutoMask2D('Largest', [], current_mask, []);
    if numel(find(current_mask)) > n_el*1.25; break; end
    handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, current_mask|original_mask, ii);
    current_mask = imerode(current_mask, se);
    n_slices_propogated = n_slices_propogated + 1;
  end
  
  current_mask = labeled_masks == jj;
  for ii=handles.SliceN-1:-1:1
    original_mask = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, ii);
    original_image = GetImage2D(handles, ii);
    current_mask = epr_AutoMask2D('FloodFillAdjusted', original_image, current_mask, safeget(handles, 'flood_threshold', 3));
    current_mask = epr_AutoMask2D('Largest', [], current_mask, []);
    if numel(find(current_mask)) > n_el*1.25, break; end
    handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, current_mask|original_mask, ii);
    current_mask = imerode(current_mask, se);
    n_slices_propogated = n_slices_propogated + 1;
  end
  disp(sprintf('Object %i: %i slices propogated', jj, n_slices_propogated));
end

guidata(handles.figure1, handles);
DrawSlice(handles);
disp('Propogation is finished.');

% --------------------------------------------------------------------
function mAutoInterpolate_Callback(hObject, ~, handles)
if length(handles.Masks) < 1
  disp('No mask is selected.');
  return;
end

nMask = get(handles.lbMasks, 'Value');
original_mask = GetImage3D(handles, handles.Masks{nMask}.Mask);

% determine the scope of interpolation
nvox = squeeze(sum(sum(original_mask, 1), 2));
slices = nvox ~= 0;
sz = size(original_mask);
max_idx = find(slices,1,'last');
min_idx = find(slices,1,'first');
if isempty(min_idx) || isempty(max_idx), return; end
applied_slices = min_idx:max_idx;
applied_slices = applied_slices(~slices(applied_slices));
if isempty(applied_slices), return; end

% calculate number of voxels expected
nvox_exp = interp1(find(slices), nvox(slices), applied_slices);

% build distance volume in available planes
interp_mask = original_mask(:,:, slices);
sz_interp = size(interp_mask);

volout=ones(sz_interp)*max([sz_interp(1) sz_interp(2)])/2;
for ns=1:sz_interp(3)
  imsk=squeeze(interp_mask(:,:,ns));
  dout=bwdist(imsk);
  din=bwdist(~imsk);
  volout(:,:,ns)=dout-din;
end

% interpolate this volume
[x,y,z]=meshgrid(1:sz(2), 1:sz(1), 1:sz(3));
new_mask = interp3(x(:,:, slices), y(:,:, slices), z(:,:, slices), volout, ...
  x(:,:, applied_slices), y(:,:, applied_slices), z(:,:, applied_slices));

% Threshold to account for expected voxels
for ii=1:length(applied_slices)
  cut_level = 1;
  the_slice = new_mask(:,:,ii);
  while numel(find(the_slice(:) < cut_level)) < nvox_exp(ii)
    cut_level=cut_level+1;
  end
  original_mask(:,:,applied_slices(ii)) = the_slice < cut_level;
end
handles.Masks{nMask}.Mask = SetMask3D(handles, original_mask);
StoreMasks(handles);
DrawSlice(handles)
disp('Interpolation is finished.');

% --------------------------------------------------------------------
function [ret1, ret2, ret3] = ginput_workaround(handles, arg1)
fld1 = get(handles.figure1, 'WindowScrollWheelFcn');
[ret1, ret2, ret3] = ginput(arg1);
set(handles.figure1, 'WindowScrollWheelFcn', fld1);

% --------------------------------------------------------------------
function [SliceN] = Position2Pixel(handles, n, pos)
SliceN = handles.SliceN;
ax = handles.ax{n}.axis(1);
ay = handles.ax{n}.axis(2);

py = (pos(:,1) - handles.ax{ay}.min)/(handles.ax{ay}.max-handles.ax{ay}.min)*(handles.ax{ay}.n-1)+1;
px = (pos(:,2) - handles.ax{ax}.min)/(handles.ax{ax}.max-handles.ax{ax}.min)*(handles.ax{ax}.n-1)+1;

SliceN(handles.ax{n}.axis) = [px,py];

% --------------------------------------------------------------------
function varargout = Pixel2Position(handles, pos)
pos(:,2) = (pos(:,2)-1)/(length(handles.ax1)-1)*(handles.box_max(1)-handles.box_min(1))+handles.box_min(1);
pos(:,1) = (pos(:,1)-1)/(length(handles.ax2)-1)*(handles.box_max(2)-handles.box_min(2))+handles.box_min(2);
if nargout == 2
  varargout{1} = pos(:,1);
  varargout{2} = pos(:,2);
else
  varargout{1} = pos;
end

% --------------------------------------------------------------------
function pbAddXYZ_Callback(hObject, ~, handles)
[ret1, ret2] = ginput_workaround(handles, 1);

switch handles.ViewDirection
  case 1,
    pos = Position2Pixel(handles,[ret1,ret2]);
    handles.XYZs{end+1}.data = fix([pos(2), handles.SliceN, pos(1)]);
  case 2
    pos = Position2Pixel(handles,[ret1,ret2]);
    handles.XYZs{end+1}.data = fix([handles.SliceN, pos(2), pos(1)]);
  case 3
    pos = Position2Pixel(handles,[ret1,ret2]);
    handles.XYZs{end+1}.data = fix([pos(1), pos(2), handles.SliceN]);
end

switch hObject
  case handles.pbAddXYZ1, the_suffix = '_XYZ1'; 
  case handles.pbAddXYZ2, the_suffix = '_XYZ2'; 
  case handles.pbAddXYZ3, the_suffix = '_XYZ3'; 
  case handles.pbAddXYZ4, the_suffix = '_XYZ4'; 
  case handles.pbAddXYZ5, the_suffix = '_XYZ5'; 
  otherwise, the_suffix = '';
end
answer = inputdlg('Choose the name', 'Name', 1, {sprintf('A%i_%i_%i%s', handles.XYZs{end}.data, the_suffix)});

if ~isempty(answer)
  handles.XYZs{end}.Name = answer{1};
  guidata(handles.figure1, handles);
  
  ListAllMasks(handles);
  DrawSlice(handles)
end

% --------------------------------------------------------------------
function pbRemoveXYZ_Callback(hObject, ~, handles)
if isempty(handles.XYZs), return; end
val = get(handles.lbXYZ , 'Value');

res = arbuz_FindImage(handles.hh, 'slave', 'Name', handles.XYZs{val}.Name, {});
arbuz_DeleteImage(handles.hh, res);
arbuz_UpdateInterface(handles.hh);

if handles.isSlave
  im2 = arbuz_FindImage(handles.hh, {struct('ImageIdx',handles.image.ImageIdx)}, '', '', {'Aslave', 'FullName', 'SlaveList'});
  find_XYZ3D     = arbuz_FindImage(handles.hh, im2{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Name', 'A', 'Color'});
else
  im2 = arbuz_FindImage(handles.hh, handles.image, '', '', {'Aslave', 'FullName', 'SlaveList'});
  find_XYZ3D    = arbuz_FindImage(handles.hh, im2{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Name', 'A', 'Color'});
end

handles.XYZs = {};
for ii = 1:length(find_XYZ3D)
  if handles.isSlave
    handles.XYZs{end+1}.data = htransform_vectors(inv(handles.image.Aslave), find_XYZ3D{ii}.data);
  else
    handles.XYZs{end+1}.data = find_XYZ3D{ii}.data;
  end
  handles.XYZs{end}.Name = find_XYZ3D{ii}.Name;
  if handles.is2D, handles.XYZs{end}.data(3) = 1; end
end
guidata(handles.figure1, handles);

ListAllMasks(handles);
DrawSlice(handles)


% --------------------------------------------------------------------
function pbInfo_Callback(hObject, ~, handles)
the_message = ['Plugin for 2D and 3D image segmentation.'];
msgbox(sprintf(the_message),'Info', 'help')

% --------------------------------------------------------------------
function mAutoKmean_Callback(hObject, ~, handles)
original_image = GetImage2D(handles, handles.SliceN);
h = figure(22); clf; set(h, 'units','normalized'); % set(h, 'outerposition',[0 0 1 1]); 
imagesc(original_image); axis image;
sz = size(original_image);

ncluster = 3;
hh = imellipse;
wait(hh);
emask = createMask(hh);
% delete(hh);

kmask = kmeans(original_image(emask),ncluster);
mask1 = zeros(sz); mask1(emask) = kmask;

imagesc(mask1); axis image;
[xpos,ypos]=ginput(1);
xpos=round(xpos);
ypos=round(ypos);
selcolor=mask1(ypos,xpos);
selmask = mask1 == selcolor;
delete(h);

% some math
selmask=bwmorph(selmask,'fill', 5);
% selmask=bwmorph(selmask,'erode', 2);
selmask=bwmorph(selmask,'erode', 2);
selmask=bwmorph(selmask,'dilate', 2);
% selmask=bwmorph(selmask,'dilate', 2);
selmask=bwmorph(selmask,'close', 2);

% selmask=bwmorph(selmask,'open', 2);
% selmask=bwmorph(selmask,'fill', 5);

nMask = get(handles.lbMasks, 'Value');
handles.Masks{nMask}.Mask = SetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, selmask, handles.SliceN);
StoreMasks(handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function pbEditMask_Callback(hObject, eventdata, handles)
% hObject    handle to pbEditMask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function lbMasks_Callback(hObject, eventdata, handles)
val = get(handles.lbMasks , 'Value');

the_Mask = handles.Masks{val}.Mask;
switch handles.ViewDirection
  case 1, trace = squeeze(any(any(the_Mask, 2), 3));
  case 2, trace = squeeze(any(any(the_Mask, 1), 3));
  case 3, trace = squeeze(any(any(the_Mask, 1), 2));
end

if ~trace(handles.SliceN)
  find_options = find(trace);
  if ~isempty(find_options)
    set(handles.eSliceN1, 'string', num2str(find_options(1)));
    handles.SliceN = SetSlice(handles, 0);
    guidata(hObject, handles);
  end
end

DrawSlice(handles)

% --------------------------------------------------------------------
function pbAutoXYZ_Callback(hObject, eventdata, handles)
val = get(handles.lbMasks , 'Value');

arg         = handles.Masks{val};
arg.Anative = handles.image.Anative;
% the_Mask = handles.Masks{val}.Mask;

res = PlaceFiducials(arg);

% Delete all markers

% Add new markers
if ~isempty(res)
  answer = inputdlg('Choose the name', 'Name', 1, {'A'});
  if ~isempty(answer)
    for ii=1:length(res)
      name1 = [num2str(res{ii}.IJK1(1)),'_',num2str(res{ii}.IJK1(2)),'_',num2str(res{ii}.IJK1(3))];
      name2 = [num2str(res{ii}.IJK2(1)),'_',num2str(res{ii}.IJK2(2)),'_',num2str(res{ii}.IJK2(3))];
      handles.XYZs{end+1}.data = res{ii}.IJK1;
      handles.XYZs{end}.Name = [answer{1},name1,'_XYZ',num2str(ii)];
      handles.XYZs{end+1}.data = res{ii}.IJK2;
      handles.XYZs{end}.Name = [answer{1},name2,'_XYZ',num2str(ii)];
    end
    
    guidata(handles.figure1, handles);
    
    ListAllMasks(handles);
    DrawSlice(handles)
  end
end

% --------------------------------------------------------------------
function mAutoTreshold_Callback(hObject, eventdata, handles)

res=inputdlg({'Low threshold:', 'High threshold:'}, 'Enter parameters',1, {'0.15', '1'});
if isempty(res), return; end

nMask = get(handles.lbMasks, 'Value');
data = GetImage3D(handles, handles.image.data);

low_threshold = str2double(res{1});
high_threshold = str2double(res{2});

switch handles.ApplyTo
  case 1
    im_mask = data >= low_threshold & data <= high_threshold;
  case 2
    im_mask = false(size(data));
    for ii=1:size(data,3)
      im_mask(:,:,ii) = data(:,:,ii) >= low_threshold & data(:,:,ii) <= high_threshold;
    end
  case 3
    im_mask = GetImage3D(handles, handles.Masks{idx}.Mask);
    im_mask(:,:,handles.SliceN) = data(:,:,handles.SliceN) >= low_threshold & data(:,:,handles.SliceN) <= high_threshold;
end

handles.Masks{nMask}.Mask = SetMask3D(handles, im_mask);
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mAutoDecimate_Callback(hObject, eventdata, handles)
nMask = get(handles.lbMasks, 'Value');

answer=inputdlg({'Select decimation step'},'Decimation',...
  1,{'10'});

if ~isempty(answer)
  n = fix(str2num(answer{1}));
  n = max(n, 2);
  N = GetSliceNumber(handles);
  
  for ii=handles.SliceN:n:N
    for jj=ii+1:ii+n-1
      switch handles.ViewDirection
        case 1, handles.Masks{nMask}.Mask(jj, :, :) = false;
        case 2, handles.Masks{nMask}.Mask(:, jj, :) = false;
        case 3, handles.Masks{nMask}.Mask(:, :, jj) = false;
      end
    end
  end
  
  StoreMasks(handles);
end

% --------------------------------------------------------------------
function mAutoClone_Callback(hObject, eventdata, handles)
nMask = get(handles.lbMasks, 'Value');

answer=inputdlg({'Select slice to clone'},'Cloning',...
  1,{'10'});

if ~isempty(answer)
  n = fix(str2num(answer{1}));
  N = GetSliceNumber(handles);
  clone = GetMask2D(handles.Masks{nMask}.Mask, handles.ViewDirection, handles.SliceN);
  
  if n > 0, 
    step = 1; max_slice = max(N, handles.SliceN+n); 
  else, step = -1;  max_slice = max(1, handles.SliceN+n); 
  end
  
  for ii=handles.SliceN:step:max_slice
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(ii, :, :) = clone;
      case 2, handles.Masks{nMask}.Mask(:, ii, :) = clone;
      case 3, handles.Masks{nMask}.Mask(:, :, ii) = clone;
    end
  end
  
  StoreMasks(handles);
end



function eSliceN2_Callback(hObject, ~, handles)
% hObject    handle to eSliceN2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of eSliceN2 as text
%        str2double(get(hObject,'String')) returns contents of eSliceN2 as a double


% --- Executes during object creation, after setting all properties.
function eSliceN2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to eSliceN2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function eSliceN3_Callback(hObject, eventdata, handles)
% hObject    handle to eSliceN3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of eSliceN3 as text
%        str2double(get(hObject,'String')) returns contents of eSliceN3 as a double


% --- Executes during object creation, after setting all properties.
function eSliceN3_CreateFcn(hObject, eventdata, handles)

if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --------------------------------------------------------------------
function mVisualClear_Callback(hObject, eventdata, handles)
ClearCustomTools(handles);

% --------------------------------------------------------------------
function handles = BuildEllipseTool(handles)

settings{1} = struct('name', 'x', 'min', 0, 'max',20, 'value', 0);
settings{2} = struct('name', 'y', 'min', 0, 'max',20, 'value', 0);
settings{3} = struct('name', 'z', 'min', 0, 'max',20, 'value', 0);
settings{4} = struct('name', 'rx', 'min', 0, 'max',20, 'value', 6);
settings{5} = struct('name', 'ry', 'min', 0, 'max',20, 'value', 6);
settings{6} = struct('name', 'rz', 'min', 0, 'max',20, 'value', 6);
settings{7} = struct('name', 'pitch', 'min', -90, 'max',90, 'value', 0);
settings{8} = struct('name', 'tilt', 'min', -90, 'max',90, 'value', 0);

col_shift = 100;
for ii=1:8
  col   = 20 - floor((ii-1)/9)*20;
  row  = 10 + rem(ii-1, 9)*col_shift;
  settings{ii}.text = uicontrol('Style','text', 'Parent', handles.pEllipse,  ...
    'Units', 'pixels', 'Position', [row, col, 25, 15], 'string', settings{ii}.name, ...
    'Callback', '');
  settings{ii}.edit = uicontrol('Style','edit', 'Parent', handles.pEllipse,  ...
    'Units', 'pixels', 'Position', [row+20, col, 55, 15], 'string', num2str(settings{ii}.value), ...
    'Callback', 'TTouchImageBasedPLG(''fse_mask'',gcbo,guidata(gcbo))');
end

handles.ellipse = settings;
guidata(handles.figure1, handles);

function fse_mask(hh, handles)
for ii=1:8
  data.(handles.ellipse{ii}.name) = str2double(handles.ellipse{ii}.edit.String);
end

[X2, X1, X3] = meshgrid(handles.ax{2}.xx, handles.ax{1}.xx, handles.ax{3}.xx);
new_mask = false(handles.ax{1}.n, handles.ax{2}.n, handles.ax{3}.n);
aa = data.tilt * pi/180;
bb = data.pitch * pi/180;
X1p = (X1-data.x)*cos(aa) + (X2-data.y)*sin(aa);
Y1p = -(X1-data.x)*sin(aa) + (X2-data.y)*cos(aa);
X1pp = (X1p-data.x)*cos(bb) + (X3-data.z)*sin(bb);
Z1pp = -(X1p-data.x)*sin(bb) + (X3-data.z)*cos(bb);
new_mask((X1pp/data.rx).^2 + (Y1p/data.ry).^2 + (Z1pp/data.rz).^2 <= 1) = true;
if isempty(handles.Masks), return; end
nMask = get(handles.lbMasks, 'Value');
handles.Masks{nMask}.Mask = new_mask;
guidata(handles.figure1, handles);
DrawSlice(handles);
