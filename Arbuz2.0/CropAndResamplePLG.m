function varargout = CropAndResamplePLG(varargin)
% CROPANDRESAMPLEPLG M-file for CropAndResamplePLG.fig
%      CROPANDRESAMPLEPLG, by itself, creates a new CROPANDRESAMPLEPLG or
%      raises the existing
%      singleton*.
%
%      H = CROPANDRESAMPLEPLG returns the handle to a new CROPANDRESAMPLEPLG or the handle to
%      the existing singleton*.
%
%      CROPANDRESAMPLEPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CROPANDRESAMPLEPLG.M with the given input arguments.
%
%      CROPANDRESAMPLEPLG('Property','Value',...) creates a new CROPANDRESAMPLEPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CropAndResamplePLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CropAndResamplePLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CropAndResamplePLG

% Last Modified by GUIDE v2.5 20-Jan-2011 16:54:33

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @CropAndResamplePLG_OpeningFcn, ...
  'gui_OutputFcn',  @CropAndResamplePLG_OutputFcn, ...
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
function CropAndResamplePLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for Image2ContourPLG
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};
handles.contours = {};
handles.Masks = {};
handles.MaskEditMode = 2;
handles.ViewDirection = 3;
handles.ApplyTo = 1;
handles.cursor = [];
handles.op_mode = 'none';
handles.draw_mode = 'none';
handles.rubber_radius = 3;
set(handles.figure1,'KeyPressFcn',@KeyPressFunction);
handles.Crop = [1,1,1;1,1,1];

% Update handles structure
guidata(hObject, handles);

% try to load selected dataset
pbLoad_Callback(hObject, [], handles)

% --------------------------------------------------------------------
function varargout = CropAndResamplePLG_OutputFcn(hObject, eventdata, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)
find_list = GetSelectedImage(handles);

handles.Masks = {};
handles.MaskBuffer = [];

if ~isempty(find_list)
  im = arbuz_FindImage(handles.hh, {find_list}, '', '', {'data', 'Aproxy', 'FullName', 'ProxyList', 'Bbox'});
  im = im{1};
  im.max = max(im.data(:));
  im.min = min(im.data(:));
  set(handles.figure1, 'Name', sprintf('CropAndResamplePLG [%s]',im.FullName));
  
  handles.Crop = [1,1,1;size(im.data)];
  set(handles.eDim, 'String', sprintf('[%s]',num2str(size(im.data))))
  set(handles.eDimFinal, 'String', sprintf('[%s]',num2str(size(im.data))))
else
  im = [];
  set(handles.figure1, 'Name', 'CropAndResamplePLG');
end

handles.image = im;
set(handles.eSliceN, 'String', num2str(size(im.data, 3)/2));
handles.SliceN = SetSlice(handles, 0);
nSlices = GetSliceNumber(handles);
set(handles.slSlice, 'Max', nSlices, 'Min', 1, 'SliderStep', min(1/(nSlices-1)*[1,10], [1,1]));
guidata(hObject, handles);

DrawSlice(handles)

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
if isempty(handles), return; end
fig_size = get(handles.figure1, 'Position');

tool_size = 185;
wborder = 12;
hborder = 6;
w_axborder = 30;
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
  panel_size(3)-2*wborder, h_selectorpanel]);

set(handles.pCustomTools, 'Position', ...
  [wborder, 2.5*hborder+2*h_buttonsize, ...
  tool_size-2*wborder, 85]);

set(handles.pbDone, 'Position', ...
  [wborder, 0.5*hborder, tool_size-2*wborder, h_buttonsize]);

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
handles.SliceN = SetSlice(handles, 0);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)

hhandles = guidata(handles.hh);

Corner1 = handles.Crop(1,:);
Corner2 = handles.Crop(2,:);

% add proxy image
name = inputdlg('Resampled image name','Image Name');

if ~isempty(name)
  NewPts      = fix(str2num(get(handles.eDim, 'string')));
  NewPtsFinal = fix(eval(get(handles.eDimFinal, 'string')));
  if length(NewPts) < 3, NewPts = NewPts(1)*ones(1, 3); end
  if length(NewPtsFinal) < 3, NewPts = NewPts(1)*ones(1, 3); end
  
  dim_order = [2,1,3];
  Aresolution = hmatrix_translate(-[1 1 1]) * hmatrix_scale((Corner2(dim_order)-Corner1(dim_order))./(NewPtsFinal(dim_order)-1)) *...
    hmatrix_translate(Corner1(dim_order));
  new_image = [];
  image_type = class(handles.image.data);
  
  if isequal(NewPts, NewPtsFinal)
    new_image.data = handles.image.data(Corner1(dim_order(2)):Corner2(dim_order(2)), Corner1(dim_order(1)):Corner2(dim_order(1)), Corner1(dim_order(3)):Corner2(dim_order(3)));
  else
    new_image.data = reslice_volume(eye(4), Aresolution, zeros(NewPtsFinal), cast(handles.image.data, 'double'), 0, 1);
    new_image.data = cast(new_image.data, image_type);
  end
  
  new_image.Name = name{1};
  new_image.isStore = 1;
  new_image.A = Aresolution;
  hhandles = arbuz_AddProxyImage(hhandles, handles.image.ImageIdx, new_image, '3DEPRI');
  
  guidata(hhandles.MainFigure,hhandles);
  arbuz_UpdateInterface(hhandles);
end

% --------------------------------------------------------------------
function rbViewDirection_Callback(hObject, eventdata, handles)
hh = [handles.rbView1, handles.rbView2, handles.rbView3];
set(hh, 'Value', 0);
set(hObject, 'Value', 1)
handles.ViewDirection = find(hh == hObject);
handles.SliceN = SetSlice(handles, 0);
guidata(hObject, handles);
nSlices = GetSliceNumber(handles);
set(handles.slSlice, 'Max', nSlices, 'Min', 1, 'SliderStep', min(1/(nSlices-1)*[1,10],[1,1]));
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
try  val = fix(str2double(get(handles.eSliceN, 'String'))); catch err, val = 1; end
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
function DrawSlice(handles)
hh = handles.axes1;
cla(hh); hold(hh, 'on');
sz_im = size(handles.image.data);

[vd,hd] = Dir2Idx(handles.ViewDirection);

image_class = class(handles.image.data);
switch handles.ViewDirection
  case 1, imagesc(squeeze(handles.image.data(handles.SliceN,:, :))', 'Parent', hh);
  case 2, imagesc(squeeze(handles.image.data(:,handles.SliceN, :)), 'Parent', hh);
  case 3, imagesc(handles.image.data(:,:, handles.SliceN), 'Parent', hh); 
end

plot(handles.Crop(1,hd)*[1,1], [1,sz_im(vd)], 'r');
plot(handles.Crop(2,hd)*[1,1], [1,sz_im(vd)], 'r');
plot([1,sz_im(hd)], handles.Crop(1,vd)*[1,1], 'r');
plot([1,sz_im(hd)], handles.Crop(2,vd)*[1,1], 'r');

opt = safeget(handles, 'opt', []);
vis_low  = safeget(opt, 'vis_low', 0);
vis_high = safeget(opt, 'vis_high', 1);
dclim = cast(handles.image.max - handles.image.min, 'double');
axis(hh, 'image');
% if diff(sz), axis(hh, 'square'); end
set(hh,'CLim',handles.image.min + cast(fix(dclim*[vis_low, vis_high]), image_class), 'YDir', 'normal', 'XDir', 'normal');

colormap bone; 

% --------------------------------------------------------------------
function KeyPressFunction(src,evnt)
handles = guidata(src);

k= evnt.Key; %k is the key that is pressed

if strcmp(k,'leftarrow'), pbPrevSlice_Callback(handles.pbPrevSlice, [], handles);
elseif strcmp(k,'rightarrow'), pbNextSlice_Callback(handles.pbNextSlice, [], handles);
end


% --------------------------------------------------------------------
function figure1_WindowButtonMotionFcn(hObject, eventdata, handles)
switch handles.op_mode
  case {'rubber', 'brush'}
    CoordXY = get(handles.axes1, 'CurrentPoint');
    CoordXY = CoordXY(1,1:2);
    handles.cursor = DrawCursor(handles.axes1, CoordXY(1), CoordXY(2), handles.rubber_radius+0.5, handles.cursor);
    guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function figure1_WindowButtonDownFcn(hObject, eventdata, handles)
bt = get(handles.figure1, 'SelectionType');

switch handles.op_mode
  case {'rubber', 'brush'}
    if strcmp(bt, 'normal')
      if isempty(handles.Masks), return; end
      nMask = get(handles.lbMasks, 'Value');
      
      im_mask = GetStandardData(handles, handles.Masks{nMask}.Mask);
      
      r = fix(handles.rubber_radius);
      r_mask = false(r*2+1);
      dim = (1:(2*r+1))' - r - 1;
      dim1 = dim(:, ones((2*r+1), 1)); %x
      dim2 = dim1'; % y
      r_mask(dim1.^2+dim2.^2 <= r^2) = true;
      
      n1 = size(im_mask);
      CoordXY = get(handles.axes1, 'CurrentPoint');
      x = floor(CoordXY(1,1) + 0.5);
      y = floor(CoordXY(1,2) + 0.5);
      rx = x + (-r:r); rx = rx(rx >=1 & rx <=n1(2));
      ry = y + (-r:r); ry = ry(ry >=1 & ry <=n1(1));
      rmx = rx - x + r + 1;
      rmy = ry - y + r + 1;
      switch handles.op_mode
        case 'rubber', im_mask(ry,rx,handles.SliceN) = im_mask(ry,rx,handles.SliceN) & ~r_mask(rmy,rmx);
        case 'brush', im_mask(ry,rx,handles.SliceN) = im_mask(ry,rx,handles.SliceN) | r_mask(rmy,rmx);
      end
      handles.Masks{nMask}.Mask = SetStandardMask(handles, im_mask);
      guidata(handles.figure1, handles);
      DrawSlice(handles)
    else
      handles.op_mode = 'none';
      if ~isempty(handles.cursor), delete(handles.cursor(ishandle(handles.cursor))); end
      handles.cursor = [];
      ClearCustomTools(handles);
    end
  case {'vis_range', 'intrv_threshold','vis_cmap'}
    ClearCustomTools(handles);
end

% --------------------------------------------------------------------
function mSpecialVisrange_Callback(hObject, eventdata, handles)
if ~strcmp(handles.op_mode, 'none'), return; end
handles=ClearCustomTools(handles);
handles.op_mode = 'vis_range';

pos = get(handles.pCustomTools, 'Position');
opt = safeget(handles, 'opt', []);
set(handles.pCustomTools, 'Title', 'Visualization ranges');
handles.op_control{1} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 10, pos(3)-20, 20], 'Value', safeget(opt, 'vis_low', 0), ...
  'Callback', 'CropAndResamplePLG(''ff_vis_range'',gcbo,guidata(gcbo))');

handles.op_control{2} = uicontrol('Style','slider', 'Parent', handles.pCustomTools, 'Min', 0, 'Max', 1, ...
  'Units', 'pixels', 'Position', [10, 40, pos(3)-20, 20], 'Value', safeget(opt, 'vis_high', 1), ...
  'Callback', 'CropAndResamplePLG(''ff_vis_range'',gcbo,guidata(gcbo))');

guidata(handles.figure1, handles);

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
  'Callback', 'CropAndResamplePLG(''ff_cmap'',gcbo,guidata(gcbo))');

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
function ff_vis_range(hObject, handles)
handles.opt.vis_low = get(handles.op_control{1}, 'Value');
handles.opt.vis_high = get(handles.op_control{2}, 'Value');
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function ff_cmap(hObject, handles)
handles.opt.colormap = get(hObject, 'Value');
guidata(handles.figure1, handles);
DrawSlice(handles)

% --------------------------------------------------------------------
function data = GetStandardData(handles, data)
switch handles.ViewDirection
  case 1, data = permute(data, [2,3,1]);
  case 2, data = permute(data, [1,3,2]);
end

% --------------------------------------------------------------------
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
if ~strcmp(handles.draw_mode, 'none'); return; end
handles.SliceN = SetSlice(handles, eventdata.VerticalScrollCount);
guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function figure1_ButtonDownFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function figure1_WindowButtonUpFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function point = pos2Dto3D(x,y,handles)

switch handles.ViewDirection
  case 1, point = [handles.SliceN, x, y];
  case 2, point = [x, handles.SliceN, y];
  case 3, point = [y, x, handles.SliceN];
end

% --------------------------------------------------------------------
function [vd,hd,od] = Dir2Idx(view_direction)
% vertical dimension
% horizontal dimension
% orthogonal dimension

switch view_direction
  case 1, vd = 3; hd = 2; od = 1;
  case 2, vd = 1; hd = 3; od = 2;
  case 3, vd = 1; hd = 2; od = 3;
end

% --------------------------------------------------------------------
function pbCropIncDecr_Callback(hObject, eventdata, handles)
[vd,hd] = Dir2Idx(handles.ViewDirection);

str = get(handles.pmStepChange, 'string');
val = get(handles.pmStepChange, 'value');
step = str2double(str{val});

switch hObject
  case handles.pbRincrease, handles.Crop(2,hd) = iff(handles.Crop(2,hd)+step <= handles.image.Box(hd), handles.Crop(2,hd)+step, handles.image.Box(hd));
  case handles.pbRdecrease, handles.Crop(2,hd) = iff(handles.Crop(2,hd)-step > 5, handles.Crop(2,hd)-step, 5);
  case handles.pbLincrease, handles.Crop(1,hd) = iff(handles.Crop(1,hd)+step <= handles.image.Box(hd)-5, handles.Crop(1,hd)+step, handles.image.Box(hd)-5);
  case handles.pbLdecrease, handles.Crop(1,hd) = iff(handles.Crop(1,hd)-step >= 1, handles.Crop(1,hd)-step, 1);
  case handles.pbTdecrease, handles.Crop(2,vd) = iff(handles.Crop(2,vd)+step <= handles.image.Box(vd), handles.Crop(2,vd)+step, handles.image.Box(vd));
  case handles.pbTincrease, handles.Crop(2,vd) = iff(handles.Crop(2,vd)-step > 5, handles.Crop(2,vd)-step, 5);
  case handles.pbBdecrease, handles.Crop(1,vd) = iff(handles.Crop(1,vd)+step <= handles.image.Box(vd)-5, handles.Crop(1,vd)+step, handles.image.Box(vd)-5);
  case handles.pbBincrease, handles.Crop(1,vd) = iff(handles.Crop(1,vd)-step >= 1, handles.Crop(1,vd)-step, 1);
  otherwise
    xy = floor(getrect + 0.5);
    handles.Crop(1,hd) = xy(1);
    handles.Crop(2,hd) = xy(1) + xy(3);
    handles.Crop(1,vd) = xy(2);
    handles.Crop(2,vd) = xy(2) + xy(4);
end

NewPts = str2num(get(handles.eDim, 'string'));
if length(NewPts) ~= 1,
  new_sz = handles.Crop(2,:) - handles.Crop(1,:) + 1;
  set(handles.eDim, 'string', sprintf('[%s]',num2str(new_sz)))
  set(handles.eDimFinal, 'string', sprintf('[%s]',num2str(new_sz)))
end

guidata(hObject, handles);
DrawSlice(handles);

% --------------------------------------------------------------------
function slSlice_Callback(hObject, eventdata, handles)
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

