function varargout = Image2ContourPLG(varargin)
% IMAGE2CONTOURPLG M-file for Image2ContourPLG.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2008

% Last Modified by GUIDE v2.5 07-Jan-2008 09:47:09

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Image2ContourPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @Image2ContourPLG_OutputFcn, ...
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
function Image2ContourPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for Image2ContourPLG
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};
handles.contours = {};

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = Image2ContourPLG_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
if isempty(handles), return; end
fig_size = get(handles.figure1, 'Position');

tool_size = 32;
wborder = 1.5;
hborder = 0.5;
w_axborder = 6;
h_axborder = 3;
h_buttonsize = 2.2;
h_buttonsize_tools = 1.8;
h_selectorpanel = 5.3;
h_auto_panel     = 16;

panel_size = [fig_size(3)-tool_size-wborder, hborder, tool_size, fig_size(4)-2*hborder];
set(handles.panelTools, 'Position',panel_size);
set(handles.axes1, 'Position', ...
  [wborder+w_axborder, hborder+h_axborder, ...
  fig_size(3)-tool_size-3*wborder-w_axborder, ...
  fig_size(4)-2*hborder-h_axborder]);

set(handles.panelImageSelector, 'Position', ...
  [wborder, panel_size(4) - h_selectorpanel - hborder, ...
  panel_size(3)-2*wborder, h_selectorpanel]);

set(handles.pbLoad, 'Position', ...
  [wborder, ...
  panel_size(4) - h_selectorpanel - 2*hborder - h_buttonsize,...
  panel_size(3)-2*wborder, h_buttonsize]);

hhs_manual = [handles.pbDrawEllipse, handles.pbDrawPoly, ...
  handles.pbClear, handles.pbManualDone];

for ii=1:length(hhs_manual)
set(hhs_manual(ii), 'Position', ...
  [wborder, panel_size(4) - 10 - ii*h_buttonsize_tools-ii*hborder/2,...
  panel_size(3)-2*wborder, h_buttonsize_tools]);
end

set(handles.panelAutoSelector, 'Position', ...
  [wborder, ...
  panel_size(4) - 36,...
  panel_size(3)-2*wborder, h_auto_panel]);
auto_panel_size = get(handles.panelAutoSelector, 'Position');

set(handles.pmColorFrame, 'Position', ...
  [wborder, auto_panel_size(4)-3-hborder, ...
  auto_panel_size(3) - 6 - 3*wborder, h_buttonsize_tools]);

set(handles.pbEyedrop, 'Position', ...
  [auto_panel_size(3) - 6 - wborder, ...
  auto_panel_size(4)-2.9-hborder, ...
  6, h_buttonsize_tools]);

set(handles.eBWvector, 'Position', ...
  [wborder, auto_panel_size(4)-3-2*hborder - h_buttonsize_tools, ...
  auto_panel_size(3)-2*wborder, h_buttonsize_tools]);

set(handles.slTreshold, 'Position', ...
  [wborder, auto_panel_size(4)-3-3*hborder - 2*h_buttonsize_tools, ...
  auto_panel_size(3)-2*wborder, h_buttonsize_tools]);

set(handles.pbShowContour, 'Position', ...
  [wborder, auto_panel_size(4)-3-4*hborder - 3*h_buttonsize_tools, ...
  auto_panel_size(3)-2*wborder, h_buttonsize_tools]);

set(handles.pbAutoContourDone, 'Position', ...
  [wborder, hborder, auto_panel_size(3)-2*wborder, h_buttonsize]);


% --------------------------------------------------------------------
function pbDrawEllipse_Callback(hObject, eventdata, handles)

handles.contours{end+1}.h = imellipse(gca, []);
name = inputdlg('Contour name','Input', 1, {''});

if ~isempty(name)
    handles.contours{end}.name = name{1};
    handles.contours{end}.Type = 'ellipse';

    % set aspect ratio to 1
    api = iptgetapi(handles.contours{end}.h);
    pos = api.getPosition(); pos(3:4) = mean(pos(3:4));
    api.setPosition(pos);
    api.setFixedAspectRatioMode(true);

    guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbManualDone_Callback(hObject, eventdata, handles)

sel_list = GetSelectedImage(handles);

if isempty(sel_list), return; end

for ii=1:length(handles.contours)
%   if ishandle(handles.contours{ii}.h)
    api = iptgetapi(handles.contours{ii}.h);
    if handles.image.SlaveIdx >= 1
      cont.A    = handles.image.Aslave;
    else
      cont.A = eye(4);
    end
    switch handles.contours{ii}.Type
      case 'ellipse'
        vtx = api.getVertices();
        plot(vtx(:,1),vtx(:,2))
        if ~isempty(handles.contours{ii}.name)
          cont.Name = handles.contours{ii}.name;
          cont.data = zeros(size(vtx,1)+1,3); cont.data(1,2)=size(vtx,1);
          cont.data(2:end,1:2) = vtx;
          arbuz_AddProxyImage(handles.hh, handles.image.ImageIdx, cont, 'CONTOUR');
        end
      case 'poly'
        pos = api.getPosition();
        plot(pos(:,1),pos(:,2))
        if ~isempty(handles.contours{ii}.name)
          cont.Name = handles.contours{ii}.name;
          cont.data = zeros(size(pos,1)+1,3); cont.data(1,2)=size(pos,1);
          cont.data(2:end,1:2) = pos;
          cont.ImageType = 'CONTOUR';
          arbuz_AddImage(handles.hh, handles.image.ImageIdx, cont');
        end
    end
    api.delete();
%   end
end
guidata(handles.figure1,handles);

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbClear_Callback(hObject, eventdata, handles)
for ii=1:length(handles.contours)
  if ishandle(handles.contours{ii}.h)
    api = iptgetapi(handles.contours{ii}.h);
    api.delete();
  end
end
handles.contours={};
guidata(handles.figure1,handles);

% --------------------------------------------------------------------
function pbDrawPoly_Callback(hObject, eventdata, handles)

handles.contours{end+1}.h = impoly(handles.axes1, []);
name = inputdlg('Contour name','Input', 1, {''});

if ~isempty(name)
  handles.contours{end}.name = name{1};
  handles.contours{end}.Type = 'poly';
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)
find_list = GetSelectedImage(handles);

if ~isempty(find_list)
  im = arbuz_FindImage(handles.hh, {find_list}, '', '', {'data', 'Aslave', 'FullName'});
  im = im{1};
  im.max = max(im.data(:));
  im.min = min(im.data(:));
  set(handles.figure1, 'Name', sprintf('Image2ContourPLG [%s]',im.FullName));
else
  im = [];
  set(handles.figure1, 'Name', 'Image2ContourPLG');
end

im.intesity = [];
im.LAB      = [];
handles.image = im;
handles.auto_contour = [];
guidata(hObject, handles);

pbClear_Callback(hObject, eventdata, handles)

DrawImage(handles)

% --------------------------------------------------------------------
function pb1_Callback(hObject, eventdata, handles)

[im, scale] = GetImage(handles);
 
figure;
%   bwselect(im)
a = bwlabel(im);

% --------------------------------------------------------------------
function slTreshold_Callback(hObject, eventdata, handles)
pbShowContour_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function eBWvector_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function cbBWImage_Callback(hObject, eventdata, handles)
DrawImage(handles)


% --------------------------------------------------------------------
function pbEyedrop_Callback(hObject, eventdata, handles)
xy = fix(ginput(1));
color_vector = double(squeeze(handles.image.data(xy(2), xy(1), :)))';
color_vector = color_vector/norm(color_vector);

set(handles.eBWvector, 'String', num2str(color_vector));
handles.image.intesity = [];
DrawImage(handles)

% --------------------------------------------------------------------
function pmColorFrame_Callback(hObject, eventdata, handles)
DrawImage(handles)

% --------------------------------------------------------------------
function pbShowContour_Callback(hObject, eventdata, handles)
[im, scale] = GetImage(handles);
delete(findobj('Type', 'hggroup'));

if isrgb(im), im = rgb2gray(im); end

pos = get(handles.slTreshold, 'Value');
% [c,h] = contour(handles.axes1, double(im),[pos 1.1].* double(scale(2)));
% set(h, 'Color', [1 1 1], 'LineWidth', 1);
% c(3,:) = 0;

treshold = pos*double(scale(2));
mask = im <= treshold;

% mask_dial  = bwmorph(mask, 'dilate',5);
% mask_bridge  = bwmorph(mask, 'bridge',4);
% mask_clean  = bwmorph(mask, 'clean');
mask_morph = bwmorph(mask, 'open');

[c,h] = contour(handles.axes1, mask_morph,[0.5 0.5]);
set(h, 'Color', [0 1 0], 'LineWidth', 1);
c(3,:) = 0;

handles.auto_contour = c';
guidata(hObject, handles);

disp('Done')
% --------------------------------------------------------------------
function pbAutoContourDone_Callback(hObject, eventdata, handles)

if ~isempty(handles.auto_contour)
  name = inputdlg('Contour name','Input', 1, {''});

  if ~isempty(name)
    cont.Name = name{1};
    cont.data = handles.auto_contour;
    if handles.image.ProxyIdx >= 1
      cont.A    = handles.image.Aproxy;
    else
      cont.A = eye(4);
    end
    arbuz_AddProxyImage(handles.hh, handles.image.ImageIdx, cont, 'CONTOUR');
    arbuz_UpdateInterface(handles.hh);
  end
end

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

if get(handles.rbSelected, 'Value')
    find_list = arbuz_FindImage(handles.hh, 'all', 'Selected', 1, {});
else
    find_list = arbuz_FindImage(handles.hh, 'all', 'Highlighted', 1, {});
end
find_list = arbuz_FindImage(handles.hh, find_list, 'ImageType', '2D', {});

if isempty(find_list)
    disp('Image2ContourPLG: No 2D image data found.');
else
    find_list = find_list{1};
end

% --------------------------------------------------------------------
function DrawImage(handles)
axes(handles.axes1);
cla

[im, scale] = GetImage(handles);

if ~isempty(im)
  switch get(handles.pmColorFrame, 'Value')
    case 2
    imagesc(im, scale); colormap gray
    otherwise
    imagesc(im, scale);
  end
  axis image;  hold on
end

% --------------------------------------------------------------------
function [im, scale] = GetImage(handles)

pos = get(handles.slTreshold, 'Value');

if ~isempty(handles.image)
  switch get(handles.pmColorFrame, 'Value')
    case 1 % raw data
      im = handles.image.data;
      scale = [handles.image.max*pos,handles.image.max];
    case 2 % intensity data
      if isempty(handles.image.intesity)
        %   color_vector = [0.2989, 0.5870, 0.1140];
        color_vector = str2num(get(handles.eBWvector, 'String'));
        color_vector = color_vector/norm(color_vector)/1.3;
        im = handles.image.data(:,:,1) * color_vector(1) + ...
          handles.image.data(:,:,2) * color_vector(2) + ...
          handles.image.data(:,:,3) * color_vector(3);
        handles.image.intesity = im;
        guidata(handles.figure1, handles);
      else
        im = handles.image.intesity;
      end
    scale = [handles.image.max*pos,handles.image.max];
    case {3,4,5,6}
      if isempty(handles.image.LAB)
        cform = makecform('srgb2lab');
        data_c = applycform(handles.image.data,cform);
        ab = double(data_c(:,:,2:3));
        nrows = size(ab,1);
        ncols = size(ab,2);
        ab = reshape(ab,nrows*ncols,2);

        nColors = 3;
        % repeat the clustering 3 times to avoid local minima
        [cluster_idx,cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
          'Replicates',3);

        pixel_labels = reshape(cluster_idx,nrows,ncols);
        rgb_label = repmat(pixel_labels,[1 1 3]);

        handles.image.LAB.LABdata = data_c;
        handles.image.LAB.idx = pixel_labels;
        
        color = data_c;
        color(rgb_label ~= 1) = 0;
        handles.image.LAB.a = color;

        color = data_c;
        color(rgb_label ~= 2) = 0;
        handles.image.LAB.b = color;

        color = data_c;
        color(rgb_label ~= 3) = 0;
        handles.image.LAB.c = color;
        guidata(handles.figure1, handles);
      end
      switch get(handles.pmColorFrame, 'Value')
        case 3, im = handles.image.LAB.idx;
        case 4, im = handles.image.LAB.a;
        case 5, im = handles.image.LAB.b;
        case 6, im = handles.image.LAB.c;
      end
    scale = [0, max(im(:))];
  end
else
  im = [];
  scale = [0,1];
end



