function varargout = GuideMaskEditPLG(varargin)
% GUIDEMASKEDITPLG M-file for GuideMaskEditPLG.fig
%      GUIDEMASKEDITPLG, by itself, creates a new GUIDEMASKEDITPLG or
%      raises the existing
%      singleton*.
%
%      H = GUIDEMASKEDITPLG returns the handle to a new GUIDEMASKEDITPLG or
%      the handle to
%      the existing singleton*.
%
%      GUIDEMASKEDITPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in GUIDEMASKEDITPLG.M with the given input arguments.
%
%      GUIDEMASKEDITPLG('Property','Value',...) creates a new
%      GUIDEMASKEDITPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before GuideMaskEditPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property
%      application
%      stop.  All inputs are passed to GuideMaskEditPLG_OpeningFcn via
%      varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help GuideMaskEditPLG

% Last Modified by GUIDE v2.5 11-Oct-2016 15:05:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @GuideMaskEditPLG_OpeningFcn, ...
  'gui_OutputFcn',  @GuideMaskEditPLG_OutputFcn, ...
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
function GuideMaskEditPLG_OpeningFcn(hObject, eventdata, handles, varargin)
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
handles.opt.vis_abs = 1;
handles.opt.colormap = 'jet';
handles.assignments = [];

set(handles.figure1,'KeyPressFcn',@KeyPressFunction);

handles.icons = load('SliceMaskEditICO.mat');

% create toolbar
TB_buttons{1, 1} = struct('icon', 'AddPoly','Callback','GuideMaskEditPLG(''mDrawAdd_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{1, 2} = struct('icon', 'RemovePoly','Callback','GuideMaskEditPLG(''mDrawErase_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{2, 1} = struct('icon', 'AddFreeHand','Callback','GuideMaskEditPLG(''mDrawAddFreehand_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{2, 2} = struct('icon', 'EraseFreeHand','Callback','GuideMaskEditPLG(''mDrawEraseFreehand_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{3, 1} = struct('icon', 'CircleBrush','Callback','GuideMaskEditPLG(''mDrawBrush_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{3, 2} = struct('icon', 'CircleEraser','Callback','GuideMaskEditPLG(''mDrawRubber_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{4, 1} = struct('icon', 'FloodFill','Callback','GuideMaskEditPLG(''mSpecialFloodFill_Callback'',gcbo,[],guidata(gcbo))');
TB_buttons{4, 2} = struct('icon', 'DeleteContour','Callback','GuideMaskEditPLG(''mDrawDelete_Callback'',gcbo,1,guidata(gcbo))');
TB_buttons{5, 1} = struct('icon', 'Pencil1','Callback','GuideMaskEditPLG(''mDrawPencil_Callback'',gcbo,1,guidata(gcbo))');
TB_buttons{6, 2} = struct('icon', 'DeleteAll','Callback','GuideMaskEditPLG(''mDrawDeleteAll_Callback'',gcbo,1,guidata(gcbo))');
TB_buttons{7, 1} = struct('icon', 'OpenImage','Callback','GuideMaskEditPLG(''mAuto_Callback'',gcbo,2,guidata(gcbo))');
TB_buttons{7, 2} = struct('icon', 'CloseImage','Callback','GuideMaskEditPLG(''mAuto_Callback'',gcbo,1,guidata(gcbo))');
TB_buttons{9, 1} = struct('icon', 'Propogate','Callback','GuideMaskEditPLG(''mAutoPropogate_Callback'',gcbo,1,guidata(gcbo))');
TB_buttons{9, 2} = struct('icon', 'Interpolate','Callback','GuideMaskEditPLG(''mAutoInterpolate_Callback'',gcbo,1,guidata(gcbo))');

for ii=1:size(TB_buttons, 1)
  for jj=1:size(TB_buttons, 2)  
    if isfield(TB_buttons{ii,jj}, 'icon')
      icon = handles.icons.(TB_buttons{ii,jj}.icon);
      handles.ToolBar{ii,jj} = uicontrol('Style', 'pushbutton', 'Parent', handles.figure1, ...
        'Units', 'pixels', 'Callback',TB_buttons{ii,jj}.Callback,...
        'CData', icon,'TooltipString',TB_buttons{ii,jj}.icon);
    else
      handles.ToolBar{ii,jj} = -1;
    end
  end
end

% Update handles structure
guidata(hObject, handles);


% try to load selected dataset
pbLoad_Callback(hObject, [], handles)

% --------------------------------------------------------------------
function varargout = GuideMaskEditPLG_OutputFcn(hObject, eventdata, handles)

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
handles.questions = {};

if ~isempty(find_list)
  im = arbuz_FindImage(handles.hh, {find_list}, '', '', {'data', 'Aslave', 'FullName', 'SlaveList', 'A', 'Box', 'Anative'});
  im = im{1};
  if strcmp(im.ImageType, '2D'), handles.is2D = 1; end
  if strcmp(im.ImageType, '3DSURFACE'), return; end
  if strcmp(im.ImageType, 'XYZ'), return; end
  % im.data(im.data > 100) = 0; % fix this
  % im.data(im.data < -10) = 0; % fix this
  
  im.max = cast(max(im.data(:)), 'double');
  im.min = cast(min(im.data(:)), 'double');
  set(handles.figure1, 'Name', sprintf('GuideMaskEditPLG [%s]',im.FullName));
  
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
  set(handles.figure1, 'Name', 'GuideMaskEditPLG');
end

handles.image = im;
handles.SliceN = SetSlice(handles, 0);

% assign slice
nSlices = GetSliceNumber(handles);
set(handles.slSlice, 'Max', nSlices, 'Min', 1, 'SliderStep', min(1/(nSlices-1)*[1,10], [1,1]));

% reset zoom parameters
[zoom, handles.box_min, handles.box_max] = GetImage2Ddims(handles);
handles.idx1 = true(zoom(1), 1);
handles.idx2 = true(zoom(2), 1);
handles.ax1 = linspace(handles.box_min(1), handles.box_max(1), zoom(1));
handles.ax2 = linspace(handles.box_min(2), handles.box_max(2), zoom(2));

handles.OldMasks{1} = handles.Masks;
handles.OldMasks{2} = handles.Masks;
guidata(hObject, handles);

ListAllMasks(handles);
set_questions(handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
if isempty(handles), return; end
fig_size = get(handles.figure1, 'Position');

left_toolbar_buttons_size = size(handles.ToolBar);
% create left toolbar
for ii=1:left_toolbar_buttons_size(1)
  for jj=1:left_toolbar_buttons_size(2)
    if ~isequal(handles.ToolBar{ii,jj}, -1)
      set(handles.ToolBar{ii,jj}, 'Position', [2+(jj-1)*30, fig_size(4) - 40 - (ii-1)*30,28,28]);
    end
  end
end

tool_size = 185;
wborder = 12;
hborder = 6;
w_axborder = 60;
h_axborder = 15;
h_buttonsize = 28;
h_selectorpanel = 40;

h_navig_panel = 60;
w_navig_button = 100;
slider_width = 15;

panel_size = [fig_size(3)-tool_size-wborder, hborder, tool_size, fig_size(4)-2*hborder];
set(handles.panelTools, 'Position',panel_size);
set(handles.axes1, 'Position', ...
  [wborder+w_axborder, hborder+h_axborder+h_navig_panel, ...
  fig_size(3)-tool_size-3*wborder-w_axborder, ...
  fig_size(4)-2*hborder-h_axborder-h_navig_panel]);

w_navig_panel = fig_size(3)-tool_size-2*w_axborder;
navig_panel_offset = (w_navig_panel - (w_navig_button+wborder)*4)/2 + w_axborder;
h_navig_element = h_navig_panel - 3*hborder - slider_width;
pos = [navig_panel_offset + wborder, 2*hborder + slider_width, w_navig_button, h_navig_panel - 3*hborder - slider_width];
set(handles.pbPrevSlice, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder) * 3, 2*hborder + slider_width, w_navig_button, h_navig_element];
set(handles.pbNextSlice, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder) * 2, 2 * hborder+slider_width, w_navig_button, h_navig_element];
set(handles.eSliceN, 'Position', pos);
pos = [navig_panel_offset + wborder + (w_navig_button+wborder),2*hborder+slider_width-1, w_navig_button, h_navig_element];
set(handles.textSliceN, 'Position', pos);
pos = [navig_panel_offset + wborder, hborder, w_navig_button * 4 + wborder*3, slider_width];
set(handles.slSlice, 'Position', pos);

set(handles.pbLoad, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - hborder, ...
  panel_size(3)-2*wborder-5-h_selectorpanel, h_selectorpanel]);

set(handles.pbInfo, 'Position', ...
  [panel_size(3)-wborder-h_selectorpanel, panel_size(4) - h_selectorpanel - hborder, ...
  h_selectorpanel, h_selectorpanel]);

set(handles.uipanel4, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - 40, ...
  panel_size(3)-2*wborder, h_selectorpanel]);

set(handles.uipanel7, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - 150, ...
  panel_size(3)-2*wborder, 106]);

set(handles.uipanelXYZ, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - 250, ...
  panel_size(3)-2*wborder, 90]);
ctls = [handles.pbAddXYZ1, handles.pbAddXYZ2, handles.pbAddXYZ3, handles.pbAddXYZ4, handles.pbAddXYZ5, handles.pbAddXYZ];
for ii = 1:6
  set(ctls(ii), 'Position', [-22 + 25*ii, 60, 25, 25])
end

l = panel_size(4) - 510;
set(handles.pQuestionPanel, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - l - 250, ...
  tool_size-2*wborder, l]);
set(handles.lbQuestion, 'position', [2, 2, 130, l-20]);


set(handles.pbAddQuestion, 'position', [134, l-40, 24, 24]);
set(handles.pbDeleteQuestion, 'position', [134, l-70, 24, 24]);
set(handles.pbNextQuestion, 'position', [134, l-100, 24, 24]);

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
function eSliceN_Callback(hObject, eventdata, handles)
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
nMask = handles.assignments.added;

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

  if ii == safeget(handles.assignments, 'inquestion', -1), continue; end
  
  new_image = [];
  new_image.data = handles.Masks{ii}.Mask;
  new_image.ImageType = '3DMASK';
  new_image.Name = handles.Masks{ii}.Name;
  new_image.A = iff(handles.image.SlaveIdx > 0, handles.image.Aslave, eye(4));
  new_image.isStore = 1;
  arbuz_AddImage(handles.hh, new_image, handles.image.Image);
end

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbSaveSurface_Callback(hObject, eventdata, handles) %#ok<DEFNU>
if isempty(handles.image), return; end
nMask = get(handles.lbMasks, 'Value');

sz = handles.image.Box;
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
[zoom, handles.box_min, handles.box_max] = GetImage2Ddims(handles);
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
function SliceN = SetSlice(handles, n)
try  val = str2double(get(handles.eSliceN, 'String')); catch err, val = 1; end
val = val + n;

N = GetSliceNumber(handles);
if val >= 1 && val <= N, SliceN = val;
elseif val < 1, SliceN = N;
else
  SliceN = 1;
end
set(handles.eSliceN, 'String', num2str(SliceN));
set(handles.slSlice, 'Value', SliceN);

% --------------------------------------------------------------------
function n = GetSliceNumber(handles)
n = size(handles.image.data, handles.ViewDirection);

% --------------------------------------------------------------------
function [n, box_min, box_max] = GetImage2Ddims(handles)
n = size(handles.image.data);
use_dim = (1:length(n))~=handles.ViewDirection;
n = n(use_dim);
box_max = [handles.image.Box, 1]*handles.image.Anative; 
box_min = [1,1,1,1]*handles.image.Anative;
box_max = box_max(use_dim);
box_min = box_min(use_dim);

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
  the_tag = safeget(handles.Masks{ii}, 'tag', '');
  switch the_tag
    case 'selected', the_tag = ' sel';
    case 'outline', the_tag = ' out'; 
  end
  
  [the_color] = arbuz_Color('', ii);
  color_structure = safeget(handles.Masks{ii}, 'Color', []); 
  the_color = safeget(color_structure, 'Color', the_color);
  if strcmp(the_color,  'blue'), the_color = [0, 0, 1]; end
  if strcmp(the_color,  'red'), the_color = [1, 0, 0]; end
  if strcmp(the_color,  'green'), the_color = [0, 1, 0]; end
  
  if isequal(the_color, [1,1,1])
    the_html_color = '000000';
  else
    the_html_color = sprintf('%02x%02x%02x', fix(the_color(1)*255), fix(the_color(2)*255), fix(the_color(3)*255));
  end
  str{end+1} = ['<html><font color=',the_html_color,'>',the_name,'</font>',the_tag,'</html>'];
end
if isempty(str), str = {'No masks'}; end
val = get(handles.lbMasks, 'Value');
set(handles.lbMasks, 'String', str, 'Value', min([val, length(str)]));


% --------------------------------------------------------------------
function DrawSlice(handles)
hh = handles.axes1;
cla(hh);

ax1 = handles.ax1(handles.idx1);
ax2 = handles.ax2(handles.idx2);
if handles.is2D
  image_slice = handles.image.data;
else
  switch handles.ViewDirection
    case 1, image_slice=squeeze(handles.image.data(handles.SliceN,handles.idx1, handles.idx2));
    case 2, image_slice=squeeze(handles.image.data(handles.idx1,handles.SliceN, handles.idx2));
    case 3, image_slice=handles.image.data(handles.idx1,handles.idx2, handles.SliceN);
  end
end

hhh = imagesc(ax2, ax1, image_slice, 'Parent', hh);
% using oxygen alpha mask
set(hhh, 'AlphaData', image_slice > -50);

axis(hh, 'equal');
axis(hh, 'image');

opt = safeget(handles, 'opt', []);
vis_abs = safeget(opt, 'vis_abs', 0);
vis_low = safeget(opt, 'vis_low', 0);
vis_high = safeget(opt, 'vis_high', 35);

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
for ii=1:length(handles.Masks)
  
  if ii == safeget(handles.assignments, 'added', -1), continue; end
%   if ii == safeget(handles.assignments, 'removed', -1), continue; end

  
  Mask = GetMask2D(handles.Masks{ii}.Mask, handles.ViewDirection, handles.SliceN);
  
  if any(Mask(:))
    legend_str{end+1} = safeget(handles.Masks{ii}, 'Name', '?');
    the_color = arbuz_Color('', ii);
    color_structure = safeget(handles.Masks{ii}, 'Color', []); 

    [~, h] = contour(ax2, ax1, Mask(handles.idx1, handles.idx2), [0.5 0.5], 'Parent', hh);
    set(h, 'LineColor', safeget(color_structure, 'Color', the_color));
    if ii==safeget(handles.assignments, 'result', 0) || ii==safeget(handles.assignments, 'inquestion', 0)
      set(h, 'Linewidth', 2.5);
    else
      set(h, 'Linewidth', 1);
    end
  end
  
  %   % Mask before and after
  %   if ii==nMask
  %     Mask = GetMask2D(handles, handles.Masks{nMask}.Mask, handles.SliceN+1);
  %     if any(Mask(:))
  %       legend_str{end+1} = [handles.Masks{ii}.Name, ' next'];
  %       contour(Mask, [0.5], safeget(handles.Masks{ii}, 'Color', 'm'))
  %     end
  %     Mask = GetMask2D(handles, handles.Masks{nMask}.Mask, handles.SliceN-1);
  %     if any(Mask(:))
  %       legend_str{end+1} = [handles.Masks{ii}.Name, ' before'];
  %       contour(Mask, [0.5], safeget(handles.Masks{ii}, 'Color', 'y'))
  %     end
  %   end
end

csize  = abs(max(ax1)-min(ax1)) / 60;
csizey = csize;
if handles.is2D, csizey = -csize; end
for ii=1:length(handles.XYZs)
  data = handles.XYZs{ii}.data;
  data(4) = 0;
  
  switch handles.ViewDirection
    case 1, xy = Pixel2Position(handles, data([3,1])); z = data(2);
    case 2, xy = Pixel2Position(handles, data([3,2])); z = data(1);
    case 3, xy = Pixel2Position(handles, data([1,2])); z = data(3);
  end
  if z == handles.SliceN
    plot(xy(1)+2*csize*[-1, 1], xy(2)+[0,0], 'r', 'Parent', hh)
    plot(xy(1)+[0,0], xy(2)+2*csize*[-1, 1], 'r', 'Parent', hh)
    
    ha = [];
    ha(end+1) = text(xy(1)+csize,...
    xy(2)+csizey,...
    handles.XYZs{ii}.Name, 'Color', [1,1,1], 'BackgroundColor', [0,0,0], ...
    'VerticalAlignment', 'bottom', 'Tag', 'Anchors', 'interpreter', 'none');
  
  elseif abs(z - handles.SliceN) < 5
    plot(xy(1)+csize*[-1, 1], xy(2)+[0,0], 'g', 'Parent', hh)
    plot(xy(1)+[0,0], xy(2)+csize*[-1, 1], 'g', 'Parent', hh)
  end  
end

if ~isempty(legend_str), legend(legend_str,'Location', 'NorthEast', 'Parent', handles.figure1, 'interpreter', 'none'); end

if handles.is2D, 
  set(handles.axes1, 'ydir', 'reverse'); 
else
  set(handles.axes1, 'ydir', 'normal');
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
if any(point < 1) || any(point > handles.image.Box), return; end

val = handles.image.data(point(1), point(2), point(3));

% --------------------------------------------------------------------
function handles = ApplySliceMask(handles, Mask, contour_mode)
if isempty(handles.Masks) || ~isfield(handles.assignments, 'target')
  return; 
end

if strcmp(contour_mode, 'add'), nMask = handles.assignments.added; 
else, nMask = handles.assignments.removed; 
end

switch handles.ViewDirection
  case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = ...
      handles.Masks{nMask}.Mask(handles.SliceN, :,:) | Mask;
  case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = ...
      handles.Masks{nMask}.Mask(:, handles.SliceN,:) | Mask;
  case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = ...
      handles.Masks{nMask}.Mask(:,:, handles.SliceN) | Mask;
end

switch contour_mode
  case 'replace'
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = Mask;
      case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = Mask;
      case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = Mask;
    end
  case 'add'
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = ...
          handles.Masks{nMask}.Mask(handles.SliceN, :,:) | Mask;
      case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = ...
          handles.Masks{nMask}.Mask(:, handles.SliceN,:) | Mask;
      case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = ...
          handles.Masks{nMask}.Mask(:,:, handles.SliceN) | Mask;
    end
  case 'erase'
    switch handles.ViewDirection
      case 1, handles.Masks{nMask}.Mask(handles.SliceN, :,:) = ...
          handles.Masks{nMask}.Mask(handles.SliceN, :,:) | Mask;
      case 2, handles.Masks{nMask}.Mask(:, handles.SliceN,:) = ...
          handles.Masks{nMask}.Mask(:, handles.SliceN,:) | Mask;
      case 3, handles.Masks{nMask}.Mask(:,:, handles.SliceN) = ...
          handles.Masks{nMask}.Mask(:,:, handles.SliceN) | Mask;
    end
end

if strcmp(contour_mode, 'add'), 
handles.Masks{handles.assignments.removed}.Mask = ...
~handles.Masks{handles.assignments.added}.Mask & handles.Masks{handles.assignments.removed}.Mask;
else
handles.Masks{handles.assignments.added}.Mask = ...
~handles.Masks{handles.assignments.removed}.Mask & handles.Masks{handles.assignments.added}.Mask;
end

% erase and add masks should not have original mask
handles = Validate(handles);

StoreMasks(handles);

function handles = Validate(handles)
% add mask should not have original mask
handles.Masks{handles.assignments.added}.Mask = ...
~handles.Masks{handles.assignments.target}.Mask & handles.Masks{handles.assignments.added}.Mask;

handles.Masks{handles.assignments.result}.Mask = ...
handles.Masks{handles.assignments.added}.Mask | ...
(handles.Masks{handles.assignments.target}.Mask &...
~handles.Masks{handles.assignments.removed}.Mask);

handles.Masks{handles.assignments.inquestion}.Mask = ...
handles.Masks{handles.assignments.inquestion}.Mask & ...
~handles.Masks{handles.assignments.result}.Mask;

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

res=inputdlg({'Low threshold:', 'High threshold:', 'Name:'}, 'Enter parameters',1, {'0.15', '1', 'All'});
if isempty(res), return; end

low_threshold = str2num(res{1});
high_threshold = str2num(res{2});
[handles,idx] = GetMask(handles, res{3}, true);

data = GetImage3D(handles, handles.image.data);

switch handles.ApplyTo
  case 1
    im_mask = data >= low_threshold*handles.image.max & data <= high_threshold*handles.image.max;
    for ii=1:size(data,3)
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open');
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close');
    end
  case 2
    im_mask = false(size(data));
    for ii=1:size(data,3)
      mmax = max(max(data(:,:,ii)));
      im_mask(:,:,ii) = data(:,:,ii) >= low_threshold*mmax & data(:,:,ii) <= high_threshold*mmax;
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open');
      im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close');
    end
  case 3
    im_mask = GetImage3D(handles, handles.Masks{idx}.Mask);
    mmax = max(max(data(:,:,handles.SliceN)));
    im_mask(:,:,handles.SliceN) = data(:,:,handles.SliceN) >= low_threshold*mmax & data(:,:,handles.SliceN) <= high_threshold*mmax;
    im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'open');
    im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'close');
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
CoordXY = get(handles.axes1, 'CurrentPoint');
CoordXY = CoordXY(1,1:2);
switch handles.op_mode
  case {'rubber', 'brush'}
    max_ax1 = max(handles.ax1(handles.idx1));
    min_ax1 = min(handles.ax1(handles.idx1));
    max_ax2 = max(handles.ax2(handles.idx2));
    min_ax2 = min(handles.ax2(handles.idx2));
    if (CoordXY(2)-handles.rubber_radius) < min_ax1, CoordXY(2) = min_ax1+handles.rubber_radius; end
    if (CoordXY(2)+handles.rubber_radius) > max_ax1, CoordXY(2) = max_ax1-handles.rubber_radius; end
    if (CoordXY(1)-handles.rubber_radius) < min_ax2, CoordXY(1) = min_ax2+handles.rubber_radius; end
    if (CoordXY(1)+handles.rubber_radius) > max_ax2, CoordXY(1) = max_ax2-handles.rubber_radius; end
    handles.cursor = DrawCursor(handles.axes1, CoordXY(1), CoordXY(2), handles.rubber_radius, handles.cursor);
    guidata(handles.figure1, handles);
end
CXY = floor(Position2Pixel(handles,CoordXY)+0.5);
[val, pos] = pos2DtoVal(handles, CXY(1), CXY(2));
if ~isempty(val)
  set(handles.txtMousePosition, 'String', sprintf('%4.2f (%3.2f)', val, val/handles.image.max));
else
  set(handles.txtMousePosition, 'String', '');
end

% --------------------------------------------------------------------
function figure1_WindowButtonUpFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
bt = get(handles.figure1, 'SelectionType');

switch handles.op_mode
  case {'rubber', 'brush'}
    if strcmp(bt, 'normal')
      if isempty(handles.Masks), return; end
      nMask = get(handles.lbMasks, 'Value');
      
      im_mask = GetImage3D(handles, handles.Masks{nMask}.Mask);
      
      r(1,:) = Position2Pixel(handles, [0,0]);
      r(2,:) = Position2Pixel(handles, handles.rubber_radius*[1,1]);
      r = fix(abs(mean(diff(r,1)))+0.5);
      r_mask = false(r*2+1);
      dim = (1:(2*r+1))' - r - 1;
      dim1 = dim(:, ones((2*r+1), 1)); %x
      dim2 = dim1'; % y
      r_mask(dim1.^2+dim2.^2 <= r^2) = true;
      
      n1 = size(im_mask);
      CoordXY = get(handles.axes1, 'CurrentPoint');
      [x, y] = Position2Pixel(handles,[CoordXY(1,1), CoordXY(1,2)]);

      x = floor(x + 0.5);
      y = floor(y + 0.5);
      rx = x + (-r:r); rx = rx(rx >=1 & rx <=n1(2));
      ry = y + (-r:r); ry = ry(ry >=1 & ry <=n1(1));
      rmx = rx - x + r + 1;
      rmy = ry - y + r + 1;
      switch handles.op_mode
        case 'rubber', im_mask(ry,rx,handles.SliceN) = im_mask(ry,rx,handles.SliceN) & ~r_mask(rmy,rmx);
        case 'brush', im_mask(ry,rx,handles.SliceN) = im_mask(ry,rx,handles.SliceN) | r_mask(rmy,rmx);
      end
      handles.Masks{nMask}.Mask = SetMask3D(handles, im_mask);
      StoreMasks(handles);
      DrawSlice(handles)
    else
      handles.op_mode = 'none';
      if ~isempty(handles.cursor), delete(handles.cursor(ishandle(handles.cursor))); end
      handles.cursor = [];
      ClearCustomTools(handles);
    end
  otherwise
    ClearCustomTools(handles);
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
  'Callback', 'GuideMaskEditPLG(''ff_rubber_size'',gcbo,guidata(gcbo))');

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
  'Callback', 'GuideMaskEditPLG(''ff_rubber_size'',gcbo,guidata(gcbo))');

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
  'Callback', 'GuideMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', setmin, 'Max', setmax, ...
  'Units', 'pixels', 'Position', [10, 50, pos(3)-20, 15], 'Value', safeget(opt, 'vis_high', 1), ...
  'Callback', 'GuideMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{3} = uicontrol('Style','checkbox', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 10, 60, 15], 'Value', safeget(opt, 'vis_abs', 0), ...
  'Callback', 'GuideMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{4} = uicontrol('Style','edit', 'Parent', handles.pCustomTools,  ...
  'Units', 'pixels', 'Position', [30, 10, 55, 15], 'string', num2str(safeget(opt, 'vis_low', 0)), ...
  'Callback', 'GuideMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');
handles.op_control{5} = uicontrol('Style','edit', 'Parent', handles.pCustomTools,  ...
  'Units', 'pixels', 'Position', [95, 10, 55, 15], 'string', num2str(safeget(opt, 'vis_low', 0)), ...
  'Callback', 'GuideMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

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
  'Callback', 'GuideMaskEditPLG(''ff_cmap'',gcbo,guidata(gcbo))');

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
  'Callback', 'GuideMaskEditPLG(''ff_intrv_threshold'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 40, pos(3)-20, 20], 'Value', safeget(opt, 'high_threshold', 1), ...
  'Callback', 'GuideMaskEditPLG(''ff_intrv_threshold'',gcbo,guidata(gcbo))');

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
    'Callback', 'GuideMaskEditPLG(''ff_flood_threshold'',gcbo,guidata(gcbo))');
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
set(handles.eSliceN, 'String', num2str(SliceN));
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
    'Callback', 'GuideMaskEditPLG(''ff_strel_size'',gcbo,guidata(gcbo))');
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
function varargout = Position2Pixel(handles, pos)
pos(:,2) = (pos(:,2) - handles.box_min(1))/(handles.box_max(1)-handles.box_min(1))*(length(handles.ax1)-1)+1;
pos(:,1) = (pos(:,1) - handles.box_min(2))/(handles.box_max(2)-handles.box_min(2))*(length(handles.ax2)-1)+1;
if nargout == 2
  varargout{1} = pos(:,1);
  varargout{2} = pos(:,2);
else
  varargout{1} = pos;
end

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

if ~isfield(handles.assignments, 'target'), return; end;

pO2 = handles.image.data;
MaskTumor = handles.Masks{handles.assignments.target}.Mask;


switch hObject
  case handles.pbAddXYZ1,

    % initial processing
    se = epr_strel('sphere', 1);
    handles.Masks{handles.assignments.added}.Mask = imdilate(MaskTumor, se);
    handles.Masks{handles.assignments.removed}.Mask = false(size(MaskTumor));
    
  case handles.pbAddXYZ2
    
    % find all adjacent voxels
    result = handles.Masks{handles.assignments.result}.Mask;
    removed = handles.Masks{handles.assignments.removed}.Mask;
     
    pquestion = (pO2 > -20 & pO2 <= 10);
    CC2 = bwconncomp(pquestion|result, 18); % 6/18/26
    
    handles.questions = {};
    for jj=1:length(CC2.PixelIdxList)
      
      idx = CC2.PixelIdxList{jj};
      
      % no overlap
      in_result  = result(idx);
      in_removed = removed(idx);
      if isempty(find(in_result, 1)), continue; end
      
      edited_idx = idx(in_result == false & in_removed == false);
      if ~isempty(edited_idx)
        mask = false(size(result));
        mask(edited_idx) = true;
        CC1 = bwconncomp(mask, 18); % 6/18/26
        
        for kk=1:length(CC1.PixelIdxList)
          handles.questions{end+1}.idx = CC1.PixelIdxList{kk};
          handles.questions{end}.name = ['b3-',num2str(length(CC1.PixelIdxList{kk}))];
        end
      end
    end
    
  case handles.pbAddXYZ3
    
    % find all adjacent voxels
    result = handles.Masks{handles.assignments.result}.Mask;
    removed = handles.Masks{handles.assignments.removed}.Mask;
     
    pquestion = (pO2 > -20 & pO2 <= 10);
    CC2 = bwconncomp(pquestion|result, 18); % 6/18/26
    
    handles.questions = {};
    for jj=1:length(CC2.PixelIdxList)
      
      idx = CC2.PixelIdxList{jj};
      
      % no overlap
      in_result  = result(idx);
      in_removed = removed(idx);
      if isempty(find(in_result, 1)), continue; end
      
      edited_idx = idx(in_result == false & in_removed == false);
      if ~isempty(edited_idx)
        mask = false(size(result));
        mask(edited_idx) = true;
        CC1 = bwconncomp(mask, 6); % 6/18/26
        
        for kk=1:length(CC1.PixelIdxList)
          handles.questions{end+1}.idx = CC1.PixelIdxList{kk};
          handles.questions{end}.name = ['b3-',num2str(length(CC1.PixelIdxList{kk}))];
        end
      end
    end
  case handles.pbAddXYZ4
    
    % find all adjacent voxels
    result = handles.Masks{handles.assignments.result}.Mask;
    removed = handles.Masks{handles.assignments.removed}.Mask;
    
    pquestion = (pO2 > -20 & pO2 <= 10);
    CC2 = bwconncomp(pquestion|result, 4); % 6/18/26
    
    handles.questions = {};
    for jj=1:length(CC2.PixelIdxList)
      
      idx = CC2.PixelIdxList{jj};
      
      % no overlap
      in_result  = result(idx);
      in_removed = removed(idx);
      if isempty(find(in_result, 1)), continue; end
      
      edited_idx = idx(in_result == false & in_removed == false);
      if ~isempty(edited_idx)
        mask = false(size(result));
        mask(edited_idx) = true;
        CC1 = bwconncomp(mask, 6); % 6/18/26
        
        for kk=1:length(CC1.PixelIdxList)
          handles.questions{end+1}.idx = CC1.PixelIdxList{kk};
          handles.questions{end}.name = ['b3-',num2str(length(CC1.PixelIdxList{kk}))];
        end
      end
    end
end

handles = Validate(handles);
guidata(handles.figure1, handles);
set_questions(handles);
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

find_slice_with_mask(handles, val);

function find_slice_with_mask(handles, val)

the_Mask = handles.Masks{val}.Mask;
switch handles.ViewDirection
  case 1, trace = squeeze(any(any(the_Mask, 2), 3));
  case 2, trace = squeeze(any(any(the_Mask, 1), 3));
  case 3, trace = squeeze(any(any(the_Mask, 1), 2));
end

if ~trace(handles.SliceN)
  find_options = find(trace);
  if ~isempty(find_options)
    set(handles.eSliceN, 'string', num2str(find_options(1)));
    handles.SliceN = SetSlice(handles, 0);
    guidata(handles.figure1, handles);
  end
end

DrawSlice(handles)

% --- Executes on button press in pushbutton26.
function pushbutton26_Callback(hObject, eventdata, handles)
val = get(handles.lbMasks , 'Value');

name = handles.Masks{val}.Name;
added   = -1;
removed = -1;
inquestion = -1;
result = -1;
for ii=1:length(handles.Masks)
  if ii == val, handles.Masks{ii}.tag = 'selected';
  elseif isequal(handles.Masks{ii}.Name,[name,'+'])
    added = ii;
  elseif isequal(handles.Masks{ii}.Name, [name,'-'])
    removed = ii;
  elseif isequal(handles.Masks{ii}.Name, [name,'?'])
    inquestion = ii;
  elseif isequal(handles.Masks{ii}.Name, [name,'_out'])
    result = ii;
  elseif isequal(safeget(handles.Masks{ii}, 'tag', ''), 'selected')
    handles.Masks{ii}.tag = '';
  end
end

if added == -1
  handles.Masks{end+1}.Mask = zeros(size(handles.Masks{val}.Mask));
  handles.Masks{end}.Name = [name,'+'];
  handles.Masks{end}.Color = [];
  added = length(handles.Masks);
end
if removed == -1
  handles.Masks{end+1}.Mask = zeros(size(handles.Masks{val}.Mask));
  handles.Masks{end}.Name = [name,'-'];
  handles.Masks{end}.Color = [];
  removed = length(handles.Masks);
end
if inquestion == -1
  handles.Masks{end+1}.Mask = zeros(size(handles.Masks{val}.Mask));
  handles.Masks{end}.Name = [name,'?'];
  inquestion = length(handles.Masks);
end
if result == -1
  handles.Masks{end+1}.Mask = handles.Masks{val}.Mask;
  handles.Masks{end}.Name = [name,'_out'];
  result = length(handles.Masks);
end

handles.Masks{result}.Color = struct('Color', [0,0,0]);
handles.Masks{inquestion}.Color = struct('Color', [1,0,1]);
handles.Masks{added}.Color = struct('Color', [0,0,0]);
handles.Masks{removed}.Color = struct('Color', [0,0,0]);

handles.assignments.target = val;
handles.assignments.added = added;
handles.assignments.inquestion = inquestion;
handles.assignments.removed = removed;
handles.assignments.result = result;

guidata(handles.figure1, handles);
ListAllMasks(handles);
DrawSlice(handles);


% --- Executes on button press in pushbutton27.
function pushbutton27_Callback(hObject, eventdata, handles)
val = get(handles.lbMasks , 'Value');

name = handles.Masks{val}.Name;
added   = -1;
removed = -1;
for ii=1:length(handles.Masks)
  if ii == val, handles.Masks{ii}.tag = 'outline';
  elseif isequal(handles.Masks{ii}.Name,[name,'+'])
    added = ii;
  elseif isequal(handles.Masks{ii}.Name, [name,'-'])
    removed = ii;
  elseif isequal(handles.Masks{ii}.Name, [name,'?'])
    inquestion = ii;
  elseif isequal(safeget(handles.Masks{ii}, 'tag', ''), 'outline')
    handles.Masks{ii}.tag = '';
  end
end
guidata(handles.figure1, handles);
ListAllMasks(handles);


% --- Executes on selection change in lbQuestion.
function lbQuestion_Callback(hObject, eventdata, handles)
idx = get(handles.lbQuestion, 'value');
handles.Masks{handles.assignments.inquestion}.Mask = ...
  false(size(handles.Masks{handles.assignments.target}.Mask));
handles.Masks{handles.assignments.inquestion}.Mask(handles.questions{idx}.idx) = true;

guidata(handles.figure1, handles);
find_slice_with_mask(handles, handles.assignments.inquestion);

% --- Executes on button press in pbAddQuestion.
function pbAddQuestion_Callback(hObject, eventdata, handles)
idx = get(handles.lbQuestion, 'value');
added = false(size(handles.Masks{handles.assignments.target}.Mask));
added(handles.questions{idx}.idx) = true;

handles.Masks{handles.assignments.added}.Mask = ...
  handles.Masks{handles.assignments.added}.Mask | added;

handles = Validate(handles);

maxidx = length(get(handles.lbQuestion, 'string'));
idx = min(idx+1,maxidx);
set(handles.lbQuestion, 'value', idx);

handles.Masks{handles.assignments.inquestion}.Mask = ...
  false(size(handles.Masks{handles.assignments.target}.Mask));
handles.Masks{handles.assignments.inquestion}.Mask(handles.questions{idx}.idx) = true;

guidata(handles.figure1, handles);
find_slice_with_mask(handles, handles.assignments.inquestion);

% --- Executes on button press in pbDeleteQuestion.
function pbDeleteQuestion_Callback(hObject, eventdata, handles)
idx = get(handles.lbQuestion, 'value');
removed = false(size(handles.Masks{handles.assignments.target}.Mask));
removed(handles.questions{idx}.idx) = true;

handles.Masks{handles.assignments.removed}.Mask = ...
  handles.Masks{handles.assignments.removed}.Mask | removed;

handles = Validate(handles);

maxidx = length(get(handles.lbQuestion, 'string'));
idx = min(idx+1,maxidx);
set(handles.lbQuestion, 'value', idx);

handles.Masks{handles.assignments.inquestion}.Mask = ...
  false(size(handles.Masks{handles.assignments.target}.Mask));
handles.Masks{handles.assignments.inquestion}.Mask(handles.questions{idx}.idx) = true;

guidata(handles.figure1, handles);
find_slice_with_mask(handles, handles.assignments.inquestion);

function set_questions(handles)
idx = get(handles.lbQuestion, 'value');

str = {};
for ii=1:length(handles.questions)
  str{end+1} = handles.questions{ii}.name;
end
idx = max(1, min(length(str), idx));
set(handles.lbQuestion, 'string', str, 'value', idx)


% --- Executes on button press in pbNextQuestion.
function pbNextQuestion_Callback(hObject, eventdata, handles)
idx = get(handles.lbQuestion, 'value');

maxidx = length(get(handles.lbQuestion, 'string'));
idx = min(idx+1,maxidx);
set(handles.lbQuestion, 'value', idx);

handles.Masks{handles.assignments.inquestion}.Mask = ...
  false(size(handles.Masks{handles.assignments.target}.Mask));
handles.Masks{handles.assignments.inquestion}.Mask(handles.questions{idx}.idx) = true;

guidata(handles.figure1, handles);
find_slice_with_mask(handles, handles.assignments.inquestion);
