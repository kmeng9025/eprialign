function varargout = SliceMaskEditPLG(varargin)
% SLICEMASKEDITPLG M-file for SliceMaskEditPLG.fig
% See also: GUIDE, GUIDATA, GUIHANDLES

% Last Modified by GUIDE v2.5 14-Feb-2022 14:23:28

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @SliceMaskEditPLG_OpeningFcn, ...
  'gui_OutputFcn',  @SliceMaskEditPLG_OutputFcn, ...
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
function SliceMaskEditPLG_OpeningFcn(hObject, ~, handles, varargin)
% Choose default command line output for Image2ContourPLG
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};
handles.contours = {};
handles.XYZs = {};
handles.ViewDirection = 3;
handles.ApplyTo = 3;
handles.cursor = [];
handles.op_mode = 'none';
handles.draw_mode = 'none';
handles.rubber_radius = 3;
handles.is2D = 0;
handles.opt.vis_abs = 0;
handles.LastChoice = [];
handles.moptions.isRestrict = true;

set(handles.figure1,'KeyPressFcn',@KeyPressFunction);

handles.icons = load('SliceMaskEditICO.mat');

% menus

handles.mDrawAdd.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''AddPoly'',guidata(gcbo))';
handles.mDrawErase.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''RemovePoly'',guidata(gcbo))';
handles.mDrawAddFreehand.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''AddFreeHand'',guidata(gcbo))';
handles.mDrawEraseFreehand.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''EraseFreeHand'',guidata(gcbo))';

handles.mAutoPropogate.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''AddPoly'',guidata(gcbo))';
handles.mDrawDelete.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''DeleteAll'',guidata(gcbo))';
handles.mDrawBrush.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''CircleBrush'',guidata(gcbo))';
handles.mDrawRubber.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''CircleEraser'',guidata(gcbo))';
handles.mSpecialFloodFill.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''FloodFill'',guidata(gcbo))';
handles.mAutoPropogate.Callback = 'SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''Interpolate'',guidata(gcbo))';

% create toolbar
TB_buttons{1, 1} = struct('icon', 'AddPoly','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''AddPoly'',guidata(gcbo))');
TB_buttons{1, 2} = struct('icon', 'RemovePoly','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''RemovePoly'',guidata(gcbo))');
TB_buttons{2, 1} = struct('icon', 'AddFreeHand','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''AddFreeHand'',guidata(gcbo))');
TB_buttons{2, 2} = struct('icon', 'EraseFreeHand','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''EraseFreeHand'',guidata(gcbo))');
TB_buttons{3, 1} = struct('icon', 'CircleBrush','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''CircleBrush'',guidata(gcbo))');
TB_buttons{3, 2} = struct('icon', 'CircleEraser','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''CircleEraser'',guidata(gcbo))');
TB_buttons{4, 1} = struct('icon', 'FloodFill','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''FloodFill'',guidata(gcbo))');
TB_buttons{4, 2} = struct('icon', 'DeleteContour','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''DeleteContour'',guidata(gcbo))');
TB_buttons{5, 1} = struct('icon', 'Pencil1','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''Pencil1'',guidata(gcbo))');
TB_buttons{6, 2} = struct('icon', 'DeleteAll','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''DeleteAll'',guidata(gcbo))');
% TB_buttons{7, 1} = struct('icon', 'OpenImage','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''OpenImage'',guidata(gcbo))');
% TB_buttons{7, 2} = struct('icon', 'CloseImage','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''CloseImage'',guidata(gcbo))');
TB_buttons{8, 1} = struct('icon', 'Largest','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''Largest'',guidata(gcbo))');
TB_buttons{8, 2} = struct('icon', 'FillHoles','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''FillHoles'',guidata(gcbo))');
TB_buttons{10, 1} = struct('icon', 'Propogate','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''Propogate'',guidata(gcbo))');
TB_buttons{10, 2} = struct('icon', 'Interpolate','Callback','SliceMaskEditPLG(''mToolbarButtoons_Callback'',gcbo,''Interpolate'',guidata(gcbo))');

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
function varargout = SliceMaskEditPLG_OutputFcn(~, ~, handles)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
find_list = GetSelectedImage(handles);

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
  set(handles.figure1, 'Name', sprintf('SliceMaskEditPLG [%s]',im.FullName));
  
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
      idx = cMask.Set(-1, find_mask3D{ii}.data);
      cMask.SetField(idx, 'Name', find_mask3D{ii}.Name);
      cMask.SetField(idx, 'Color', find_mask3D{ii}.Color);
    else
      fprintf('Mask ''%s'' has different resolution and was not loaded.\n', find_mask3D{ii}.Name);
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
  set(handles.figure1, 'Name', 'SliceMaskEditPLG');
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

guidata(hObject, handles);

ListAllMasks(handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_ResizeFcn(~, ~, handles)
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
  [wborder, panel_size(4) - h_selectorpanel - 2*hborder - 264, ...
  panel_size(3)-2*wborder, 110]);
ctls = [handles.pbAddXYZ1, handles.pbAddXYZ2, handles.pbAddXYZ3, handles.pbAddXYZ4, handles.pbAddXYZ5, handles.pbAddXYZ];
for ii = 1:6
  set(ctls(ii), 'Position', [-22 + 25*ii, 80, 25, 25])
end
set(handles.lbXYZ, 'position', [2,2,100,76])

set(handles.pCustomTools, 'Position', ...
  [wborder, 2.5*hborder+2*h_buttonsize, ...
  tool_size-2*wborder, 85]);

set(handles.pbDone, 'Position', ...
  [wborder, 1.5*hborder+h_buttonsize, tool_size-2*wborder, h_buttonsize]);

set(handles.pbSaveSurface, 'Position', ...
  [wborder, hborder, tool_size-2*wborder, h_buttonsize]);

% --------------------------------------------------------------------
function pbPrevSlice_Callback(hObject, ~, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, -1);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbNextSlice_Callback(hObject, ~, handles) 
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, +1);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function eSliceN_Callback(hObject, ~, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, 0);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mEditUndo_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
cMask.Restore();
ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbAddMask_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
val = get(handles.pmMaskType, 'Value');
str = get(handles.pmMaskType, 'String');
pos = cMask.Set(-1, false(size(handles.image.data)));
cMask.SetField(pos, 'Name', str{val});
cMask.SetField(pos, 'Color', []);
ListAllMasks(handles);
set(handles.lbMasks, 'Value', cMask.n());

% --------------------------------------------------------------------
function pbDeleteMask_Callback(~, ~, handles)
val = get(handles.lbMasks, 'Value');
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
cMask.Delete(val);
ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbMaskColor_Callback(~, ~, ~)

% --------------------------------------------------------------------
function DrawPolyContour(handles, contour_mode)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');
if ~cMask.is(nMask), return; end

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

sz = size(cMask.Get(nMask));
switch handles.ViewDirection
  case 1, Mask = poly2mask(pos(:,1), pos(:,2), sz(2), sz(3));
  case 2, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(3));
  case 3, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(2));
end

handles.draw_mode = 'none';
guidata(handles.figure1, handles)
cMask.Apply2D(nMask, Mask, handles.ViewDirection, handles.SliceN, contour_mode);

ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function DrawFreehandContour(handles, contour_mode)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');
if ~cMask.is(nMask), return; end

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
sz = size(cMask.Get(nMask));
switch handles.ViewDirection
  case 1, Mask = poly2mask(pos(:,1), pos(:,2), sz(2), sz(3));
  case 2, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(3));
  case 3, Mask = poly2mask(pos(:,1), pos(:,2), sz(1), sz(2));
end

handles.draw_mode = 'none';
guidata(handles.figure1, handles);
cMask.Apply2D(nMask, Mask, handles.ViewDirection, handles.SliceN, contour_mode);

ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbDone_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

for ii=1:cMask.n()
  new_image = [];
  new_image.data = cMask.Get(ii);
  new_image.ImageType = '3DMASK';
  new_image.Name = cMask.GetField(ii, 'Name', '');
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
function pbSaveSurface_Callback(~, ~, handles)
if isempty(handles.image), return; end
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

nMask = get(handles.lbMasks, 'Value');

sz = handles.image.Box(1:3);
res = inputdlg({'Surface Name:', 'Max voxels in any dimension'}, 'Set Mask Name', 1, {cMask.GetField(nMask, 'Name', ''), num2str(max(sz))});

if ~isempty(res{1})
  sz_factor = fix(sz ./ [str2double(res{2}), str2double(res{2}), str2double(res{2})]);
  sz_factor(sz_factor < 1) = 1;
  im_mask = cMask.Get(nMask);
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
      cMask = contourc(compressed_mask, [0.5, 0.5]);
      Ascale = hmatrix_translate(-[1 1 1]) * hmatrix_scale(sz_factor([2,1,3])) * hmatrix_translate([1 1 1]+(sz_factor-[1 1 1])/2);
    else
      cMask = contourc(double(im_mask), [0.5, 0.5]);
    end
    cMask(3,:) = 0;
    
    cont.data = cMask';
    cont.A = Ascale*iff(handles.image.SlaveIdx > 0, handles.image.Aslave, eye(4));
    cont.ImageType = 'CONTOUR';
    cont.isLoaded = 1;
    arbuz_AddImage(handles.hh, cont, handles.image.Image);
    
    
%     figure(100); clf; %imagesc(im_mask);
%     subplot(3,1,1); imagesc(im_mask); axis image
%     subplot(3,1,2); imagesc(compressed_mask); axis image
%     subplot(3,1,3);
%     arbuz_DrawContour2D(gca, struct('data', cMask'), eye(4));
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
function pbCopy_Callback(hObject, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
if ~cMask.is(get(handles.lbMasks, 'Value')), return; end
nMask = get(handles.lbMasks, 'Value');
handles.MaskBuffer = cMask.Get2D(nMask, handles.ViewDirection, handles.SliceN);
guidata(hObject, handles);

% --------------------------------------------------------------------
function pbPaste_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
if ~cMask.is(get(handles.lbMasks, 'Value')) || isempty(handles.MaskBuffer), return; end
cMask.Set2D(get(handles.lbMasks, 'Value'), handles.MaskBuffer, handles.ViewDirection, handles.SliceN);
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
% SliceMaskEditPLG('rbApplyAll_Callback',hObject,eventdata,guidata(hObject))
function rbApplyAll_Callback(hObject, ~, handles)
hh1 = [handles.rbApplyAll, handles.rbApplyAllNorm, handles.rbApplySlice];
hh2 = [handles.tbAll, handles.tbAllN, handles.tbSlice];

if ~isempty(find(hh1 == hObject, 1))
  handles.ApplyTo = find(hh1 == hObject);
elseif ~isempty(find(hh2 == hObject, 1))
  handles.ApplyTo = find(hh2 == hObject);
end

set(hh1, 'Value', 0);
set(hh1(handles.ApplyTo), 'Value', 1)
set(hh2, 'State', 'off');
set(hh2(handles.ApplyTo), 'State', 'on')

% set(hh2, 'Value', 0);
% set(hh2(handles.ApplyTo), 'Value', 1)

guidata(hObject, handles);

% --------------------------------------------------------------------
function rbViewDirection_Callback(hObject, ~, handles)
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
box_max = [handles.image.Box(1:3), 1]*handles.image.Anative; 
box_min = [1,1,1,1]*handles.image.Anative;
box_max = box_max(use_dim);
box_min = box_min(use_dim);

% --------------------------------------------------------------------
function idx = GetMask(handles, mask_name, create_new_mask)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

new_dims = [];
if create_new_mask
  new_dims = size(handles.image.data);
end
n = cMask.n();
idx = cMask.GetIdx(mask_name, new_dims);
if ~isempty(idx) && idx > n
  ListAllMasks(handles);  
end

% --------------------------------------------------------------------
function ListAllMasks(handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

str = {};
for ii=1:cMask.n()
  the_name  = cMask.GetField(ii, 'Name', '');
  [the_color] = arbuz_Color('', ii);
  color_structure = cMask.GetField(ii, 'Color', []); 
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
if isempty(str), str = {'No anchors'}; end
val = get(handles.lbXYZ, 'Value');
set(handles.lbXYZ, 'String', str, 'Value', min([val, length(str)]), 'ListboxTop', 1);

% --------------------------------------------------------------------
function DrawSlice(handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
hh = handles.axes1;
cla(hh);

ax1 = handles.ax1(handles.idx1);
ax2 = handles.ax2(handles.idx2);
if handles.is2D
  imagesc(ax2, ax1, handles.image.data, 'Parent', hh);
else
  switch handles.ViewDirection
    case 1, imagesc(ax2, ax1, squeeze(handles.image.data(handles.SliceN,handles.idx1, handles.idx2)), 'Parent', hh);
    case 2, imagesc(ax2, ax1, squeeze(handles.image.data(handles.idx1,handles.SliceN, handles.idx2)), 'Parent', hh);
    case 3, imagesc(ax2, ax1, handles.image.data(handles.idx1,handles.idx2, handles.SliceN), 'Parent', hh);
  end
end

axis(hh, 'equal');
axis(hh, 'image');

opt = safeget(handles, 'opt', []);
vis_abs = safeget(opt, 'vis_abs', 0);
vis_low = safeget(opt, 'vis_low', 0);
vis_high = safeget(opt, 'vis_high', 1);

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
for ii=1:cMask.n()
  Mask = cMask.Get2D(ii, handles.ViewDirection, handles.SliceN);
  
  if any(Mask(:))
    legend_str{end+1} = cMask.GetField(ii, 'Name', '?');
    the_color = arbuz_Color('', ii);
    color_structure = cMask.GetField(ii, 'Color', []); 

    [~, h] = contour(ax2, ax1, Mask(handles.idx1, handles.idx2), [0.5 0.5], 'Parent', hh);
    set(h, 'LineColor', safeget(color_structure, 'Color', the_color));
    if ii==nMask
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
if safeget(handles.moptions, 'isShowAnchors', 0)
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
end

if ~isempty(legend_str), legend(legend_str,'Location', 'NorthEast', 'Parent', handles.figure1); end

if handles.is2D 
  set(handles.axes1, 'ydir', 'reverse'); 
else
  set(handles.axes1, 'ydir', 'normal');
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
function KeyPressFunction(src,evnt)
handles = guidata(src);

k= evnt.Key; %k is the key that is pressed

if strcmp(k,'leftarrow'), pbPrevSlice_Callback(handles.pbPrevSlice, [], handles);
elseif strcmp(k,'rightarrow'), pbNextSlice_Callback(handles.pbNextSlice, [], handles);
end

% --------------------------------------------------------------------
function mSpecialOutline0p15_Callback(~, ~, handles)
idx = GetMask(handles, 'Outline', true);
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

data = GetImage3D(handles, handles.image.data);

res=inputdlg({'Threshold'}, 'Enter parameters',1, {'0.15'});
threshold = str2double(res{1});

switch handles.ApplyTo
  case 1, im_mask = outside_mask3(double(data), threshold);
  case 2, im_mask = outside_mask3(double(data), threshold);
    for ii=1:size(data,3)
      im_mask(:,:,ii) = outside_mask3(data(:,:,ii), threshold);
    end
  case 3
    im_mask = cMask.GetOriented(idx,handles.ViewDirection);
    im_mask(:,:,handles.SliceN) = outside_mask3(data(:,:,handles.SliceN), str2double(res{1}));
end

cMask.SetOriented(idx, im_mask, handles.ViewDirection);
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mSpecialSelectAll_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
res=inputdlg({'Low threshold:', 'High threshold:', 'Morphology open/close/clean:', 'Name:'}, ...
  'Enter parameters',1, {'0.15', '1', 'clean', 'All'});
if isempty(res), return; end

low_threshold = str2double(res{1});
high_threshold = str2double(res{2});
morph_protocol = res{3};
morph_open = contains(morph_protocol, 'open');
morph_close = contains(morph_protocol, 'close');
morph_clean = contains(morph_protocol, 'clean');
idx = GetMask(handles, res{4}, true);

data = GetImage3D(handles, handles.image.data);

switch handles.ApplyTo
  case 1
    im_mask = data >= low_threshold*handles.image.max & data <= high_threshold*handles.image.max;
    for ii=1:size(data,3)
      if morph_open, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open'); end
      if morph_close, im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close'); end
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
    im_mask = cMask.GetOriented(idx, handles.ViewDirection);
    mmax = max(max(data(:,:,handles.SliceN)));
    im_mask(:,:,handles.SliceN) = data(:,:,handles.SliceN) >= low_threshold*mmax & data(:,:,handles.SliceN) <= high_threshold*mmax;
    if morph_open, im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'open'); end
    if morph_close, im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'close'); end
      if morph_clean, im_mask(:,:,handles.SliceN) = bwmorph(im_mask(:,:,handles.SliceN), 'clean'); end
end

cMask.SetOriented(idx,im_mask, handles.ViewDirection);
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function mSpecialFiducials_Callback(~, ~, handles)
idx = GetMask(handles, 'FID', true);
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

data = GetImage3D(handles, handles.image.data);

res=inputdlg({'Threshold:', 'Exclude mask:'}, 'Enter parameters',1, {'0.15','Outline'});
if isempty(res), return; end

threshold = str2double(res{1});
exclude_mask_name = res{2};

idx1 = GetMask(handles, exclude_mask_name, false);
if isempty(idx1)
  exclude_mask = outside_mask3(data, threshold);
else
  exclude_mask = cMask.GetOriented(idx1, handles.ViewDirection);
end

im_mask = data > threshold*max(data(:));
im_mask(exclude_mask) = false;

for ii=1:size(data,3)
  im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'open');
  im_mask(:,:,ii) = bwmorph(im_mask(:,:,ii), 'close');
end

im_mask(exclude_mask) = false;

cMask.SetOriented(idx, im_mask, handles.ViewDirection);
DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_WindowButtonMotionFcn(~, ~, handles)
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
[val, ~] = pos2DtoVal(handles, CXY(1), CXY(2));
if ~isempty(val)
  set(handles.txtMousePosition, 'String', sprintf('%4.2f (%3.2f)', val, val/handles.image.max));
else
  set(handles.txtMousePosition, 'String', '');
end

% --------------------------------------------------------------------
function figure1_WindowButtonUpFcn(~, ~, handles)

% --------------------------------------------------------------------
function figure1_WindowButtonDownFcn(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
bt = get(handles.figure1, 'SelectionType');

switch handles.op_mode
  case {'rubber', 'brush'}
    if strcmp(bt, 'normal')
      if ~cMask.are(), return; end
      nMask = get(handles.lbMasks, 'Value');
      
      im_mask = cMask.GetOriented(nMask, handles.ViewDirection);
      
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
      cMask.SetOriented(nMask, im_mask, handles.ViewDirection);
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
function mSpecialVisrange_Callback(~, ~, handles)
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
  'Callback', 'SliceMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', setmin, 'Max', setmax, ...
  'Units', 'pixels', 'Position', [10, 50, pos(3)-20, 15], 'Value', safeget(opt, 'vis_high', 1), ...
  'Callback', 'SliceMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{3} = uicontrol('Style','checkbox', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 10, 60, 15], 'Value', safeget(opt, 'vis_abs', 0), ...
  'Callback', 'SliceMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{4} = uicontrol('Style','edit', 'Parent', handles.pCustomTools,  ...
  'Units', 'pixels', 'Position', [30, 10, 55, 15], 'string', num2str(safeget(opt, 'vis_low', 0)), ...
  'Callback', 'SliceMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');
handles.op_control{5} = uicontrol('Style','edit', 'Parent', handles.pCustomTools,  ...
  'Units', 'pixels', 'Position', [95, 10, 55, 15], 'string', num2str(safeget(opt, 'vis_low', 0)), ...
  'Callback', 'SliceMaskEditPLG(''ff_vis_range'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);
ff_vis_range(handles.op_control{1}, handles);

% --------------------------------------------------------------------
function mVisualColormap_Callback(~, ~, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'vis_cmap';

pos = get(handles.pCustomTools, 'Position');
opt = safeget(handles, 'opt', []);
set(handles.pCustomTools, 'Title', 'Select colormap');
handles.op_control{1} = uicontrol('Style','popupmenu', 'Parent', handles.pCustomTools, ...
  'String', {'jet', 'bone', 'hot', 'copper', 'pink'}, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', safeget(opt, 'colormap', 2), ...
  'Callback', 'SliceMaskEditPLG(''ff_cmap'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mSpecialInteractiveThreshold_Callback(~, ~, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'intrv_threshold';

pos = get(handles.pCustomTools, 'Position');
opt = safeget(handles, 'opt', []);
set(handles.pCustomTools, 'Title', 'Interactive thresholds');
handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', safeget(opt, 'low_threshold', 0), ...
  'Callback', 'SliceMaskEditPLG(''ff_intrv_threshold'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 40, pos(3)-20, 20], 'Value', safeget(opt, 'high_threshold', 1), ...
  'Callback', 'SliceMaskEditPLG(''ff_intrv_threshold'',gcbo,guidata(gcbo))');

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
function ff_intrv_threshold(~, handles)
handles.opt.low_threshold = get(handles.op_control{1}, 'Value');
handles.opt.high_threshold = get(handles.op_control{2}, 'Value');
nMask = get(handles.lbMasks, 'Value');

dmax = handles.image.max - handles.image.min;

cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

switch handles.ApplyTo
  case 1
    tmin = handles.image.min + handles.opt.low_threshold * dmax;
    tmax = handles.image.min + handles.opt.high_threshold * dmax;
    im_mask = handles.image.data >= tmin & handles.image.data <= tmax;
    cMask.Set(nMask, im_mask);
    cMask.Math1s(nMask,nMask,handles.ViewDirection,[],'OPEN',struct('shape', 'sphere','layers',1));
    cMask.Math1s(nMask,nMask,handles.ViewDirection,[],'CLOSE',struct('shape', 'sphere','layers',1));
  case 2
    data = cMask.Oriented(handles.image.data, handles.ViewDirection);
    im_mask = false(size(data));
    for ii=1:size(im_mask,3)
      mmax = max(max(max(data(:,:,ii))), handles.image.max*0.1);
      im_mask(:,:,ii) = data(:,:,ii) >= handles.opt.low_threshold*mmax & data(:,:,ii) <= handles.opt.high_threshold*mmax;
    end
    cMask.SetOriented(nMask, im_mask, handles.ViewDirection);
    cMask.Math1s(nMask,nMask,handles.ViewDirection,[],'OPEN',struct('shape', 'sphere','layers',1));
    cMask.Math1s(nMask,nMask,handles.ViewDirection,[],'CLOSE',struct('shape', 'sphere','layers',1));
  case 3
    data = cMask.Oriented(handles.image.data, handles.ViewDirection);
    mmax = max(max(data(:,:,handles.SliceN)));
    im_mask(:,:,handles.SliceN) = data(:,:,handles.SliceN) >= handles.opt.low_threshold*mmax & data(:,:,handles.SliceN) <= handles.opt.high_threshold*mmax;
    cMask.SetOriented(nMask, im_mask, handles.ViewDirection);
    cMask.Math1s(nMask,nMask,handles.ViewDirection,handles.SliceN,'OPEN',struct('shape', 'sphere','layers',1));
    cMask.Math1s(nMask,nMask,handles.ViewDirection,handles.SliceN,'CLOSE',struct('shape', 'sphere','layers',1));
end

DrawSlice(handles)

% --------------------------------------------------------------------
function mSpecialRemoveOutsideOutline_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');

idx = GetMask(handles, 'Outline', false);
if isempty(idx)
  im_mask = cMask.GetOriented(nMask, handles.ViewDirection);
  im_mask = im_mask & outside_mask3(GetImage3D(handles, handles.image.data), 0.15);
  cMask.SetOriented(nMask, im_mask, handles.ViewDirection);
else
  cMask.Set(nMask, cMask.Get(nMask) & cMask.Get(idx));
end

DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, eventdata.VerticalScrollCount);
guidata(hObject, handles);
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
[ret1, ret2] = ginput_workaround(handles, 1);

switch handles.ViewDirection
  case 1
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
function pbRemoveXYZ_Callback(~, ~, handles)
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
function pbInfo_Callback(~, ~, ~)
the_message = 'Plugin for 2D and 3D image segmentation.';
msgbox(sprintf(the_message),'Info', 'help')

% --------------------------------------------------------------------
function mAutoKmean_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
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
cMask.Set2D(nMask, selmask, handles.ViewDirection, handles.SliceN);
DrawSlice(handles)

% --------------------------------------------------------------------
function pbEditMask_Callback(~, ~, handles)
% hObject    handle to pbEditMask (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function lbMasks_Callback(hObject, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
val = get(handles.lbMasks , 'Value');

the_Mask = cMask.Get(val);
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
    guidata(hObject, handles);
  end
end

DrawSlice(handles)

% --------------------------------------------------------------------
function pbAutoXYZ_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
val = get(handles.lbMasks , 'Value');

arg         = cMask.Get(val);
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
function mAutoTreshold_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');
data = GetImage3D(handles, handles.image.data);

switch handles.ApplyTo
  case {1,2}
    immax = max(data(:));
  case 3
    dslice = data(:,:,handles.SliceN);
    immax = max(dslice(:));
end
res=inputdlg({'Low threshold:', 'High threshold:'}, 'Enter parameters',1, {'0.15', '1'});
if isempty(res), return; end

low_threshold = str2double(res{1}) * immax;
high_threshold = str2double(res{2}) * immax;

switch handles.ApplyTo
  case 1
    im_mask = data >= low_threshold & data <= high_threshold;
  case 2
    im_mask = false(size(data));
    for ii=1:size(data,3)
      im_mask(:,:,ii) = data(:,:,ii) >= low_threshold & data(:,:,ii) <= high_threshold;
    end
  case 3
    im_mask = cMask.GetOriented(nMask, handles.ViewDirection);
    dslice = data(:,:,handles.SliceN);
    im_mask(:,:,handles.SliceN) = dslice >= low_threshold & dslice <= high_threshold;
end

cMask.SetOriented(nMask, im_mask, handles.ViewDirection);
DrawSlice(handles)

% --------------------------------------------------------------------
function mAutoDecimate_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');

answer=inputdlg({'Select decimation step'},'Decimation',...
  1,{'10'});

if ~isempty(answer)
  n = fix(str2num(answer{1}));
  n = max(n, 2);
  N = GetSliceNumber(handles);
  
  for ii=handles.SliceN+1:n:N
    for jj=ii:ii+n-2
      cMask.Math1s(nMask, nMask, handles.ViewDirection, jj, 'ERASE', []); 
    end
  end
end

% --------------------------------------------------------------------
function mAutoClone_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');

answer=inputdlg({'Select slice to clone'},'Cloning',...
  1,{'10'});

if ~isempty(answer)
  n = fix(str2num(answer{1}));
  N = GetSliceNumber(handles);
  clone = cMask.Get2D(nMask, handles.ViewDirection, handles.SliceN);
  
  if n > 0 
    step = 1; max_slice = max(N, handles.SliceN+n); 
  else, step = -1;  max_slice = max(1, handles.SliceN+n); 
  end
  
  for ii=handles.SliceN:step:max_slice
    cMask.Set2D(nMask, nMask, clone, handles.ViewDirection, ii);
  end
end

% --------------------------------------------------------------------
function mMathXYswap_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
nMask = get(handles.lbMasks, 'Value');

sz = size(cMask.Get(nMask));

switch handles.ViewDirection
  case 1, n1 = 2; n2 = 3; swap = [1,3,2];
  case 2, n1 = 1; n2 = 3; swap = [3,2,1];
  case 3, n1 = 1; n2 = 2; swap = [2,1,3];
end

if sz(n1) == sz(n2)
  cMask.Set(nMask, permute(cMask.Get(nMask), swap));
end

DrawSlice(handles);

% --------------------------------------------------------------------
function mRestrictArea_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks', []);

iRestrict = -1;
for ii=1:cMask.n()
  if contains(cMask.GetField(ii, 'Name', ''), 'Restrict')
    iRestrict = ii; break;
  end
end

a = handles.axes1;
waitforbuttonpress
[x1, y1] = Position2Pixel(handles,a.CurrentPoint(1, 1:2));
FinalRect = rbbox;                      % return figure units
[x2, y2] = Position2Pixel(handles,a.CurrentPoint(1, 1:2));

rangex = round(min(x1,x2)):round(max(x1,x2));
rangey = round(min(y1,y2)):round(max(y1,y2));

if iRestrict == -1
  iRestrict = cMask.Set(-1, false(size(handles.image.data)));
  cMask.SetField(iRestrict, 'Name', 'Restrict');
end

newMask = false(size(handles.image.data));
 
switch handles.ViewDirection
  case 1, newMask(:, rangey, rangex) = true;
  case 2, newMask(rangey, :, rangex) = true;
  case 3, newMask(rangey, rangex,:) = true;
end

oldMask = cMask.Get(iRestrict);
if any(oldMask(:))
  cMask.Set(iRestrict, newMask & oldMask);
else
  cMask.Set(iRestrict, newMask);
end

ListAllMasks(handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function mMath2_Callback(~, ~, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
if cMask.are()
  MaskList = cMask.MaskNames();
  
  definputs(1).name = 'Operand 1';
  definputs(1).type = 3; % choice by N
  definputs(1).choices = MaskList;
  definputs(2).name = 'Operand 2';
  definputs(2).type = 3; % choice by N
  definputs(2).choices = MaskList;
  definputs(3).name = 'Operator';
  definputs(3).type = 2; % choice
  definputs(3).choices = {'AND', 'OR', 'NOTAND'};
  definputs(4).name = 'Result';
  definputs(4).type = 3; % choice by N
  definputs(4).choices = MaskList;
  
  a  = choiceinputdlg('Logical Operations',definputs);
  if ~isempty(a)
    in1 = a{1}; in2 = a{2}; selection = a{3}; out = a{4};
    switch handles.ApplyTo
      case 1, cMask.Math2(in1, in2, out, selection, []);
      case 2, cMask.Math2s(in1, in2, out, handles.ViewDirection, [], selection, []);
      case 3, cMask.Math2s(in1, in2, out, handles.ViewDirection, handles.SliceN, selection, []);
    end
  end
DrawSlice(handles);
end

% --------------------------------------------------------------------
function mMetric2_Callback(~, ~, handles)
nMask = get(handles.lbMasks, 'Value');
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
if cMask.are()
  MaskList = cMask.MaskNames();
  
  definputs(1).name = 'Operand 1';
  definputs(1).type = 3; % choice by N
  definputs(1).default = nMask; 
  definputs(1).choices = MaskList;
  definputs(2).name = 'Operand 2';
  definputs(2).type = 3; % choice by N
  definputs(2).choices = MaskList;
  definputs(3).name = 'Operator';
  definputs(3).type = 2; % choice
  definputs(3).choices = cMask.GetFunctionArguments('Metric2');

  a  = choiceinputdlg('Metrics',definputs);
  if ~isempty(a)
    in1 = a{1}; in2 = a{2}; selection = a{3};
    res = cMask.Metric2(in1, in2, selection, []);
    assignin('base', selection, res);
    fprintf('Variable %s with result is created.\n', selection);
    if isstruct(res)
      disp(selection)
      disp(res);
    else
      fprintf('%s = %f\n', selection, cMask.Metric2(in1, in2, selection, []));
    end
  end
end

% --------------------------------------------------------------------
function mMath1_Callback(~, ~, handles)
nMask = get(handles.lbMasks, 'Value');
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);
if cMask.are()
  MaskList = cMask.MaskNames();
  
  definputs(1).name = 'Operand 1';
  definputs(1).type = 3; % choice by N
  definputs(1).default = nMask; 
  definputs(1).choices = MaskList;
  definputs(2).name = 'Operator';
  definputs(2).type = 2; % choice
  switch handles.ApplyTo
      case 1, definputs(2).choices = cMask.GetFunctionArguments('Math1');
      case 2, definputs(2).choices = cMask.GetFunctionArguments('Math1s');
      case 3, definputs(2).choices = cMask.GetFunctionArguments('Math1s');
  end
  definputs(3).name = 'Result';
  definputs(3).type = 3; % choice by N
  definputs(3).choices = MaskList;
  definputs(3).default = nMask; 
  definputs(4).name = 'Option 1';
  definputs(4).type = 0; 
  definputs(4).choices = 1;
  definputs(4).default = 1; 
  
  a  = choiceinputdlg('Logical Operations',definputs);
  if ~isempty(a)
    in1 = a{1}; selection = a{2}; out = a{3}; opt1 = a{4};
    options = [];
    switch selection
      case {'ERODE', 'DILATE', 'OPEN', 'CLOSE'}
        options.layers = opt1;
      case {'LARGEST'}
        options.nLargest = opt1;
    end
    switch handles.ApplyTo
      case 1, cMask.Math1(in1, out, selection, options);
      case 2, cMask.Math1s(in1, out, handles.ViewDirection, [], selection, options);
      case 3, cMask.Math1s(in1, out, handles.ViewDirection, handles.SliceN, selection, options);
    end
  end
  DrawSlice(handles);
end

% --------------------------------------------------------------------
function mToolbarButtoons_Callback(~, func, handles)
cMask = classMaskStorage(handles.figure1, 'Masks' ,handles.moptions);

if ~cMask.are(), error('No mask present.'); end
nMask = get(handles.lbMasks, 'Value');
if ~cMask.is(nMask), error('No mask selected.'); end

handles.LastChoice.func = func;
guidata(handles.figure1, handles);

switch func
  case 'AddPoly' %----------------------------------------------------------
    handles = ModeVisualizer(handles, 'AddPoly');
    DrawPolyContour(handles, 'add');
  case 'RemovePoly' %-------------------------------------------------------
    handles = ModeVisualizer(handles, 'RemovePoly');
    DrawPolyContour(handles, 'erase');
  case 'AddFreeHand' %-------------------------------------------------------
    handles = ModeVisualizer(handles, 'AddFreehand');
    DrawFreehandContour(handles, 'add');
  case 'EraseFreeHand' %-------------------------------------------------------
    handles = ModeVisualizer(handles, 'EraseFreehand');
    DrawFreehandContour(handles, 'erase');
  case 'CircleBrush' %-------------------------------------------------------
    if ~strcmp(handles.op_mode, 'none'), return; end
    handles=ClearCustomTools(handles);
    handles.op_mode = 'brush';
    
    pos = get(handles.pCustomTools, 'Position');
    set(handles.pCustomTools, 'Title', 'Brush radius');
    handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', .2, 'Max', 5, ...
      'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', handles.rubber_radius, ...
      'Callback', 'SliceMaskEditPLG(''ff_rubber_size'',gcbo,guidata(gcbo))');
    
    guidata(handles.figure1, handles);
  case 'CircleEraser' %-------------------------------------------------------
    if ~strcmp(handles.op_mode, 'none'), return; end
    handles=ClearCustomTools(handles);
    
    handles.op_mode = 'rubber';
    
    pos = get(handles.pCustomTools, 'Position');
    set(handles.pCustomTools, 'Title', 'Rubber radius');
    handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0.2, 'Max', 5, ...
      'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', handles.rubber_radius, ...
      'Callback', 'SliceMaskEditPLG(''ff_rubber_size'',gcbo,guidata(gcbo))');
    
    guidata(handles.figure1, handles);
  case 'FloodFill' %-------------------------------------------------------
    flood_threshold = safeget(handles, 'flood_threshold', 3);
    if ~strcmp(handles.op_mode, 'ff_flood_threshold')
      handles=ClearCustomTools(handles);
      handles.op_mode = 'ff_flood_threshold';
      
      pos = get(handles.pCustomTools, 'Position');
      set(handles.pCustomTools, 'Title', 'Flood threshold [std]');
      handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 4, ...
        'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', flood_threshold, ...
        'Callback', 'SliceMaskEditPLG(''ff_flood_threshold'',gcbo,guidata(gcbo))');
    end
    
    [x, y] = ginput_workaround(handles, 1);
    [x, y] = Position2Pixel(handles,[x, y]);
    
    nMask = get(handles.lbMasks, 'Value');
    
    original_image = GetImage2D(handles, handles.SliceN);
    current_mask = cMask.Get2D(nMask, handles.ViewDirection, handles.SliceN);
    seed_mask = false(size(original_image));
    seed_mask(fix(y), fix(x)) = true;
    if any(seed_mask(:)&current_mask(:)), seed_mask = current_mask; end
    out_mask = epr_AutoMask2D('FloodFillAdjusted', original_image, seed_mask, flood_threshold);
    out_mask = epr_AutoMask2D('Largest', [], out_mask, []);
    out_mask = out_mask | current_mask;
    cMask.Set2D(nMask, out_mask, handles.ViewDirection, handles.SliceN);
    
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
    
    DrawSlice(handles);
  case 'DeleteContour' %-------------------------------------------------------
    im_mask = cMask.Get2D(nMask, handles.ViewDirection, handles.SliceN);
    
    [x, y] = ginput_workaround(handles, 1);
    [x, y] = Position2Pixel(handles,[x, y]);
    objects = bwlabel(im_mask);
    remove_mask_idx = objects(fix(y), fix(x));
    im_mask(objects == remove_mask_idx) = false;
    
    cMask.Set2D(nMask, im_mask, handles.ViewDirection, handles.SliceN);
    DrawSlice(handles)
  case 'Pencil1' %---------------------------------------------------------
    [x, y] = ginput_workaround(handles, 1);
    [x, y] = Position2Pixel(handles,[x, y]);
    
    Mask = cMask.Get2D(nMask, handles.ViewDirection, handles.SliceN);
    Mask(round(y), round(x))=true;
    cMask.Apply2D(nMask, Mask, handles.ViewDirection, handles.SliceN, 'add');
    DrawSlice(handles);
  case 'DeleteAll' %-------------------------------------------------------    
    switch handles.ApplyTo
      case 1, cMask.Math1(nMask, nMask, 'ERASE', []);
      case 2, cMask.Math1s(nMask, nMask, handles.ViewDirection, [], 'ERASE', []);
      case 3, cMask.Math1s(nMask, nMask, handles.ViewDirection, handles.SliceN, 'ERASE', []);
    end
    DrawSlice(handles)
  case 'OpenImage' %--------------------------------------------------------
    switch handles.ApplyTo
      case 1, cMask.Math1(nMask, nMask, 'OPEN', []);
      case 2, cMask.Math1s(nMask, nMask, handles.ViewDirection, [], 'OPEN', []);
      case 3, cMask.Math1s(nMask, nMask, handles.ViewDirection, handles.SliceN, 'OPEN', []);
    end
    DrawSlice(handles)
  case 'CloseImage' %-------------------------------------------------------
    switch handles.ApplyTo
      case 1, cMask.Math1(nMask, nMask, 'CLOSE', []);
      case 2, cMask.Math1s(nMask, nMask, handles.ViewDirection, [], 'CLOSE', []);
      case 3, cMask.Math1s(nMask, nMask, handles.ViewDirection, handles.SliceN, 'CLOSE', []);
    end
    DrawSlice(handles)
  case 'Largest' %-------------------------------------------------------
    switch handles.ApplyTo
      case 1, cMask.Math1(nMask, nMask, 'LARGEST', []);
      case 2, cMask.Math1s(nMask, nMask, handles.ViewDirection, [], 'LARGEST', []);
      case 3, cMask.Math1s(nMask, nMask, handles.ViewDirection, handles.SliceN, 'LARGEST', []);
    end
    DrawSlice(handles)
  case 'FillHoles' %-------------------------------------------------------
    switch handles.ApplyTo
      case 1, cMask.Math1(nMask, nMask, 'FILL-HOLES', []);
      case 2, cMask.Math1s(nMask, nMask, handles.ViewDirection, [], 'FILL-HOLES', []);
      case 3, cMask.Math1s(nMask, nMask, handles.ViewDirection, handles.SliceN, 'FILL-HOLES', []);
    end
    DrawSlice(handles)
  case 'Propogate' %--------------------------------------------------------
    se = strel('disk', 1);
    
    % split current_mask into objects
    [labeled_masks, n_labeled_masks] = bwlabel(cMask.Get2D(nMask, handles.ViewDirection, handles.SliceN));
    
    for jj=1:n_labeled_masks
      current_mask = labeled_masks == jj;
      n_el = numel(find(current_mask));
      n_slices_propogated = 0;
      for ii=handles.SliceN+1:GetSliceNumber(handles)
        original_mask = cMask.Get2D(nMask, handles.ViewDirection, ii);
        original_image = GetImage2D(handles, ii);
        current_mask = epr_AutoMask2D('FloodFillAdjusted', original_image, current_mask, safeget(handles, 'flood_threshold', 3));
        current_mask = epr_AutoMask2D('Largest', [], current_mask, []);
        if numel(find(current_mask)) > n_el*1.25; break; end
        cMask.Set2D(nMask, current_mask|original_mask, handles.ViewDirection, ii, 'replace');
        current_mask = imerode(current_mask, se);
        n_slices_propogated = n_slices_propogated + 1;
      end
      
      current_mask = labeled_masks == jj;
      for ii=handles.SliceN-1:-1:1
        original_mask = cMask.Get2D(nMask, handles.ViewDirection, ii);
        original_image = GetImage2D(handles, ii);
        current_mask = epr_AutoMask2D('FloodFillAdjusted', original_image, current_mask, safeget(handles, 'flood_threshold', 3));
        current_mask = epr_AutoMask2D('Largest', [], current_mask, []);
        if numel(find(current_mask)) > n_el*1.25, break; end
        cMask.Set2D(nMask, current_mask|original_mask, handles.ViewDirection, ii);
        current_mask = imerode(current_mask, se);
        n_slices_propogated = n_slices_propogated + 1;
      end
      fprintf('Object %i: %i slices propogated\n', jj, n_slices_propogated);
    end
    
    guidata(handles.figure1, handles);
    DrawSlice(handles);
    disp('Propogation is finished.');
  case 'Interpolate' %-------------------------------------------------------
    original_mask = cMask.GetOriented(nMask, handles.ViewDirection);
    
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
    cMask.SetOriented(nMask, original_mask, handles.ViewDirection);
    DrawSlice(handles)
    disp('Interpolation is finished.');
end

% --------------------------------------------------------------------
function mUseRestrictArea_Callback(~, ~, handles)
if contains(get(handles.mUseRestrictArea, 'Checked'), 'on')
  handles.mUseRestrictArea.Checked = 'off';
  handles.moptions.isRestrict = false;
else
  handles.mUseRestrictArea.Checked = 'on';
  handles.moptions.isRestrict = true;
end
guidata(handles.figure1, handles);


% --------------------------------------------------------------------
function mEditRepeat_Callback(hObject, ~, handles)

func = safeget(handles.LastChoice, 'func', '');
if ~isempty(func)
  mToolbarButtoons_Callback(hObject, func, handles);
end

% --------------------------------------------------------------------
function mVisualShowAnchors_Callback(hObject, eventdata, handles)
handles.moptions.isShowAnchors = contains(get(handles.mVisualShowAnchors, 'Checked'), 'on');
handles.moptions.isShowAnchors = ~handles.moptions.isShowAnchors;
set(handles.mVisualShowAnchors, 'Checked', iff(handles.moptions.isShowAnchors, 'on', 'off'))
guidata(handles.figure1, handles);
DrawSlice(handles)
