function varargout = SpecialObjectsPLG(varargin)
% SPECIALOBJECTSPLG M-file for SpecialObjectsPLG.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2008

% Last Modified by GUIDE v2.5 06-Sep-2017 13:20:03

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SpecialObjectsPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @SpecialObjectsPLG_OutputFcn, ...
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
function SpecialObjectsPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for SpecialObjectsPLG
handles.output = hObject;
handles.hh = varargin{1};

% Update handles structure
guidata(hObject, handles);

ReloadImages(hObject);
cameratoolbar(hObject);

% --------------------------------------------------------------------
function ReloadImages(hObject)
handles = guidata(hObject);
im = arbuz_FindImage(handles.hh, 'master', '','',{'NAME'});
if ~isempty(im)
  str = cell(length(im),1);
  for ii=1:length(im)
    str{ii}=im{ii}.Name;
  end
  set(handles.pmImage, 'string', str,'value',1)
else
  str = {'Images were not found'};
end

% --------------------------------------------------------------------
function varargout = SpecialObjectsPLG_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
fig_size = get(handles.figure1, 'Position');

tool_size = 28;
wborder = 1.5;
hborder = 0.5;
w_axborder = 6;
h_axborder = 2;
h_buttonsize = 2.2;
h_editsize = 1.8;

h_control_size  = 1.8;
h_control_space = 0.3;

panel_size = [fig_size(3)-tool_size-wborder, hborder, tool_size, fig_size(4)-2*hborder];
set(handles.panelTools, 'Position',panel_size);

set(handles.axes1, 'Position', ...
  [wborder+w_axborder, hborder+h_axborder, ...
  fig_size(3)-tool_size-3*wborder-w_axborder, ...
  fig_size(4)-2*hborder-h_axborder]);

control_lines = panel_size(4)-(0:9)*(h_control_size + h_control_space) - h_control_size - hborder;

set(handles.pmImage, 'Position', ...
  [wborder, control_lines(1), panel_size(3)-2*wborder, h_editsize]);

set(handles.pmObjectType, 'Position', ...
  [wborder, control_lines(2), panel_size(3)-2*wborder, h_editsize]);

set(handles.pmXYZ, 'Position', ...
  [wborder, control_lines(3), panel_size(3)-2*wborder, h_editsize]);

set(handles.pmTurn, 'Position', ...
  [wborder, control_lines(4), panel_size(3)-2*wborder, h_editsize]);

set(handles.eObjectSize, 'Position', ...
  [panel_size(3)/2, control_lines(5), panel_size(3)/2-wborder, h_editsize]);

set(handles.tVoxels, 'Position', ...
  [wborder, control_lines(5)-.5, panel_size(3)/2-wborder, h_editsize]);

set(handles.cbBringToEPR, 'Position', ...
  [wborder, control_lines(6), panel_size(3)-2*wborder, h_buttonsize]);

set(handles.pbMakeObject, 'Position', ...
  [wborder, control_lines(7), panel_size(3)-2*wborder, h_buttonsize]);

set(handles.pbDone, 'Position', ...
  [wborder, hborder, panel_size(3)-2*wborder, h_buttonsize]);

% --------------------------------------------------------------------
function pbMakeObject_Callback(hObject, eventdata, handles)
switch get(handles.pmTurn,'value')
  case 2, A = hmatrix_rotate_z(90);
  case 3, A = hmatrix_rotate_z(180);
  case 4, A = hmatrix_rotate_z(270);
  otherwise, A = eye(4);
end
switch get(handles.pmXYZ,'value')
  case 1, A = A*hmatrix_rotate_x(90);
  case 2, A = A*hmatrix_rotate_y(90);
  case 4, A = A*hmatrix_rotate_x(-90);
  case 5, A = A*hmatrix_rotate_y(-90);
  case 6, A = A;
end

switch get(handles.pmObjectType, 'Value')
  case 1
    [handles.phantom, handles.ph_opt] = cast16mm(str2double(get(handles.eObjectSize, 'String')));
    [handles.face,handles.vert]=isosurface(handles.phantom, 0.95);
    disp(sprintf('Image is generated. Memory: %g kBytes', (numel(handles.face) + numel(handles.vert))*8/1024))
  case 2
    [handles.phantom, handles.ph_opt] = cast19mm(str2double(get(handles.eObjectSize, 'String')));
    [handles.face,handles.vert]=isosurface(handles.phantom, 0.95);
    disp(sprintf('Image is generated. Memory: %g kBytes', (numel(handles.face) + numel(handles.vert))*8/1024))
  case 3
    [handles.phantom.data, handles.ph_opt] = LGR16mm(30);
    disp(sprintf('Image is generated. Memory: %g kBytes', numel(handles.phantom.data)*8/1024))
  case 4
    [handles.phantom.data, handles.ph_opt] = LGR19mm(30);
    disp(sprintf('Image is generated. Memory: %g kBytes', numel(handles.phantom.data)*8/1024))
  case 5 % 16 LGR
    [handles.phantom.data, handles.ph_opt] = MakeLGR(30, [15.91, 15.2], A);
    disp(sprintf('Image is generated. Memory: %g kBytes', numel(handles.phantom.data)*8/1024))
  case 6 % 19 LGR
    [handles.phantom.data, handles.ph_opt] = MakeLGR(30, [19.1, 15.2], A);
    disp(sprintf('Image is generated. Memory: %g kBytes', numel(handles.phantom.data)*8/1024))
  case 7 % 51 LGR
    [handles.phantom.data, handles.ph_opt] = MakeLGR(30, [51, 54], A);
    disp(sprintf('Image is generated. Memory: %g kBytes', numel(handles.phantom.data)*8/1024))
  case 8 % 4 mm ball phantom
  image_name = handles.pmImage.String{handles.pmImage.Value};
  im = arbuz_FindImage(handles.hh, 'master', 'Name', image_name,{'Name','Anative','box','grid'});

    handles.ph_opt.ImageType = '3DSURFACE';
    handles.ph_opt.sz_mm = 28;
    handles.phantom = zeros(size(im{1}.gridX));
    for ii=-1:1, for jj=-1:1, for kk=-1.5:1.5
        handles.phantom((im{1}.gridX-ii*4).^2 + (im{1}.gridY-jj*4).^2 + (im{1}.gridZ-kk*4).^2 <= 1.6) = 1;
    end, end, end
    [handles.face,handles.vert]=isosurface(handles.phantom, 0.95);
    disp(sprintf('Image is generated. Memory: %g kBytes', (numel(handles.face) + numel(handles.vert))*8/1024))
end

cla(handles.axes1)
switch handles.ph_opt.ImageType
  case '3DSURFACE'
    patch('faces',handles.face,'vertices',handles.vert,...
      'facecolor', 'none','facelighting','phong',...
      'edgecolor','black',...
      'FaceAlpha',0.3,'facevertexCdata',[0.68 0.47 0],'BackFaceLighting','unlit');
  case 'CONTOUR'
    arbuz_DrawContour3D(handles.axes1, handles.phantom, eye(4));
    xlabel('x');ylabel('y');zlabel('z');
end
guidata(hObject, handles);


% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles)

name = inputdlg('Object name','Input', 1, {''});

if ~isempty(name)
  new_image.Name = name{1};
  new_image.isLoaded = 1;
  new_image.isStore = 1;
  image_name = handles.pmImage.String{handles.pmImage.Value};
  im = arbuz_FindImage(handles.hh, 'master', 'Name', image_name,{'Name','Anative'});
  switch handles.ph_opt.ImageType
    case '3DSURFACE'
      new_image.data.face = handles.face;
      new_image.data.vert = handles.vert;
      new_image.ImageType = '3DSURFACE';
      new_image.Anative = hmatrix_translate(-handles.ph_opt.sz_vox/2*[1,1,1])*...
        hmatrix_scale(handles.ph_opt.sz_mm/handles.ph_opt.sz_vox*[1,1,1]);
      if get(handles.cbBringToEPR, 'Value')
        new_image.Anative = new_image.Anative * hmatrix_rotate_z(90) * hmatrix_rotate_y(-90);
      end
      arbuz_AddImage(handles.hh, new_image, im{1}.Name);
    case 'CONTOUR'
      new_image.data = handles.phantom.data;
      new_image.ImageType = 'CONTOUR';
      new_image.Anative = eye(4);
      if get(handles.cbBringToEPR, 'Value')
        new_image.Anative = new_image.Anative * hmatrix_rotate_z(90) * hmatrix_rotate_y(-90);
      end
      image_name = handles.pmImage.String{handles.pmImage.Value};
      im = arbuz_FindImage(handles.hh, 'master', 'Name', image_name,{'Name','Anative'});
      new_image.A = inv(im{1}.Anative);
      arbuz_AddImage(handles.hh, new_image, im{1}.Name);
  end
  
  arbuz_UpdateInterface(handles.hh);
end


% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ----- S E R V I C E     F U N C T I O N S --------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function [phantom, opt] = cast19mm(sz)

opt.ImageType = '3DSURFACE';
sz_mm = 30; % in mm
scale = sz/sz_mm;   % [voxel/mm]

phantom = zeros(sz,sz,sz);

dim = [1:sz]';
dim1 = dim(:, ones(sz, 1), ones(sz, 1)); %x
dim2 = permute(dim1, [2,1,3]); %z
dim3 = permute(dim1, [2,3,1]); %y

h_axis   = sz/2;
l_offset = sz/4;

rad_cast_rim = 22/2*scale;
l_cast_rim = l_offset - 3*scale;
r_cast_rim = l_offset + 0*scale;

rad_cast = 18.5/2*scale;
l_cast_cyl = r_cast_rim;
r_cast_cyl = l_offset + 12.92*scale;

% Here the scale differes from the real one
% to show better other edge of the cavity
rad_cast_cone = 8.7/2*scale;
l_cast_cone = r_cast_cyl;
r_cast_cone = l_offset + 15.2*scale;

rad_cast_tip = rad_cast_cone;
l_cast_tip = r_cast_cone;
r_cast_tip = l_offset + 18.915*scale;

% make rim (cyllinder)
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < rad_cast^2 & ...
  dim3 <= r_cast_cyl & dim3 >= l_cast_cyl) = 1;
% make cast body (cyllinder)
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < rad_cast_rim^2 & ...
  dim3 <= r_cast_rim & dim3 >= l_cast_rim) = 1;
% make cast cone (cone)
dy = rad_cast_cone - rad_cast;
dx = r_cast_cone - l_cast_cone;
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < ...
  (rad_cast + (dim3-l_cast_cone)*dy/dx).^2 & ...
  dim3 <= r_cast_cone & dim3 >= l_cast_cone) = 1;
% make cast tip (cyllinder)
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < rad_cast_tip^2 & ...
  dim3 <= r_cast_tip & dim3 >= l_cast_tip) = 1;

opt.sz_mm  = sz_mm;
opt.sz_vox = sz;

% --------------------------------------------------------------------
function [phantom, opt] = cast16mm(sz)

opt.ImageType = '3DSURFACE';

sz_mm = 28; % in mm
scale = sz/sz_mm;   % [voxel/mm]

from_surface0_cut1 = ([0, 1.93, 5,8.36,11.62,15.02,18.48,21.94,24.87] - 2.77)*scale;
from_surface0_cut2 = ([0.01 ,2.77,5.94,9.26,12.53,15.99,19.43, 22.83,24.86] - 2.77)*scale;

phantom = zeros(sz,sz,sz);

dim = [1:sz]';
dim1 = dim(:, ones(sz, 1), ones(sz, 1)); %x
dim2 = permute(dim1, [2,1,3]); %z
dim3 = permute(dim1, [2,3,1]); %y

h_axis   = sz/2;
l_offset = sz/4;

rad_cast_rim = 20/2*scale;
l_cast_rim = l_offset - 3*scale;
r_cast_rim = l_offset + 0*scale;

rad_cast = 15.475/2*scale;
l_cast_cyl = r_cast_rim;
r_cast_cyl = l_offset + 14*scale;

% Here the scale differes from the real one
% to show better other edge of the cavity
rad_cast_cone = 8.7/2*scale;
l_cast_cone = r_cast_cyl;
r_cast_cone = l_offset + 15.2*scale;

rad_cast_tip = rad_cast_cone;
l_cast_tip = r_cast_cone;
r_cast_tip = l_offset + 19.33*scale;

% make rim (cyllinder)
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < rad_cast^2 & ...
  dim3 <= r_cast_cyl & dim3 >= l_cast_cyl) = 1;
% make cast body (cyllinder)
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < rad_cast_rim^2 & ...
  dim3 <= r_cast_rim & dim3 >= l_cast_rim) = 1;
% make cast cone (cone)
dy = rad_cast_cone - rad_cast;
dx = r_cast_cone - l_cast_cone;
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < ...
  (rad_cast + (dim3-l_cast_cone)*dy/dx).^2 & ...
  dim3 <= r_cast_cone & dim3 >= l_cast_cone) = 1;
% make cast tip (cyllinder)
phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < rad_cast_tip^2 & ...
  dim3 <= r_cast_tip & dim3 >= l_cast_tip) = 1;

% make cast cuts
% for ii=3:length(from_surface0_cut1)-3
%   phantom((dim1-h_axis).^2+(dim2-h_axis).^2 < (rad_cast_rim+2)^2 & ...
%     (dim1-h_axis).^2+(dim2-h_axis).^2 >= rad_cast_rim^2 & ...
%     dim3 <= (from_surface0_cut2(ii)+l_offset) & ...
%     dim3 >= (from_surface0_cut1(ii)+l_offset)) = 1;
% end

opt.sz_mm  = sz_mm;
opt.sz_vox = sz;

% --------------------------------------------------------------------
function [phantom, opt] = LGR16mm(nphi)

surface0_first_cut_mid_cut = [ 2.7, 6.04, 9.305, 12.735];
% diameter
dd = 15.91;
% length
ll = 15.2;


opt.ImageType = 'CONTOUR';

face1 = zeros(nphi,3);
phi = linspace(0, 2*pi, nphi)';
face1(:, 1) = dd/2*cos(phi);
face1(:, 2) = dd/2*sin(phi);
face1(:, 3) = -ll/2;
face2 = face1;
face2(:, 3) = +ll/2;

phantom = [];
phantom(1, :) = [0, nphi, 0];
phantom(end+1:end+nphi, :) = face1;
phantom(end+1, :) = [0, nphi, 0];
phantom(end+1:end+nphi, :) = face2;
phantom(end+1, :) = [0, 2, 0];
phantom(end+1:end+2, :) = [dd/2, 0, -ll/2; dd/2, 0, +ll/2];

face1(:, 1) = (dd/2-1)*cos(phi);
face1(:, 2) = (dd/2-1)*sin(phi);
for ii=1:length(surface0_first_cut_mid_cut)
  face1(:, 3) = +ll/2 - surface0_first_cut_mid_cut(ii);
  phantom(end+1, :) = [0, nphi, 0];
  phantom(end+1:end+nphi, :) = face1;
end

phi = linspace(0, 2*pi, 9)';
for ii=1:length(phi)
  phantom(end+1, :) = [0, 2, 0];
  phantom(end+1:end+2, :) = [dd/2*cos(phi(ii)), dd/2*sin(phi(ii)), +ll/2; ...
    dd/2*cos(phi(ii)), dd/2*sin(phi(ii)), +ll/2 - 1];
end


% --------------------------------------------------------------------
function [phantom, opt] = LGR19mm(nphi)

surface0_first_cut_mid_cut = [2.8675,6.1225,9.5575,12.807];
% diameter
dd = 19.13;
% length
ll = 15.2;


opt.ImageType = 'CONTOUR';

face1 = zeros(nphi,3);
phi = linspace(0, 2*pi, nphi)';
face1(:, 1) = dd/2*cos(phi);
face1(:, 2) = dd/2*sin(phi);
face1(:, 3) = -ll/2;
face2 = face1;
face2(:, 3) = +ll/2;

phantom = [];
phantom(1, :) = [0, nphi, 0];
phantom(end+1:end+nphi, :) = face1;
phantom(end+1, :) = [0, nphi, 0];
phantom(end+1:end+nphi, :) = face2;
phantom(end+1, :) = [0, 2, 0];
phantom(end+1:end+2, :) = [dd/2, 0, -ll/2; dd/2, 0, +ll/2];

face1(:, 1) = (dd/2-1)*cos(phi);
face1(:, 2) = (dd/2-1)*sin(phi);
for ii=1:length(surface0_first_cut_mid_cut)
  face1(:, 3) = +ll/2 - surface0_first_cut_mid_cut(ii);
  phantom(end+1, :) = [0, nphi, 0];
  phantom(end+1:end+nphi, :) = face1;
end

phi = linspace(0, 2*pi, 9)';
for ii=1:length(phi)
  phantom(end+1, :) = [0, 2, 0];
  phantom(end+1:end+2, :) = [dd/2*cos(phi(ii)), dd/2*sin(phi(ii)), +ll/2; ...
    dd/2*cos(phi(ii)), dd/2*sin(phi(ii)), +ll/2 - 1];
end

% --------------------------------------------------------------------
function [phantom, opt] = MakeLGR(nphi, res_dim, direction)

% diameter
dd = res_dim(1);
% length
ll = res_dim(2);

opt.ImageType = 'CONTOUR';

face1 = zeros(nphi,3);
phi = linspace(0, 2*pi, nphi)';
face1(:, 1) = dd/2*cos(phi);
face1(:, 2) = dd/2*sin(phi);
face1(:, 3) = -ll/2;
face2 = face1;
face2(:, 3) = +ll/2;

phantom = [];
phantom(1, :) = [0, nphi, 0];
phantom(end+1:end+nphi, :) = face1;
phantom(end+1, :) = [0, nphi, 0];
phantom(end+1:end+nphi, :) = face2;
phantom(end+1, :) = [0, 2, 0];
phantom(end+1:end+2, :) = [dd/2, 0, -ll/2; dd/2, 0, +ll/2];

% face1(:, 1) = (dd/2-1)*cos(phi);
% face1(:, 2) = (dd/2-1)*sin(phi);
% for ii=1:length(surface0_first_cut_mid_cut)
%   face1(:, 3) = +ll/2 - surface0_first_cut_mid_cut(ii);
%   phantom(end+1, :) = [0, nphi, 0];
%   phantom(end+1:end+nphi, :) = face1;
% end

phi = linspace(0, 2*pi, 9)';
for ii=1:length(phi)
  phantom(end+1, :) = [0, 2, 0];
  phantom(end+1:end+2, :) = [dd/2*cos(phi(ii)), dd/2*sin(phi(ii)), +ll/2; ...
    dd/2*cos(phi(ii)), dd/2*sin(phi(ii)), +ll/2 - 1];
end

% rotate image
ii = 1;
while ii < size(phantom,1)
  p = phantom(ii,2); r1 = phantom(ii+1:ii+p,:); 
  r1 = htransform_vectors(direction, r1);
  phantom(ii+1:ii+p,:) = r1; 
  ii = ii+1+p;
end

% --------------------------------------------------------------------
function cbBringToEPR_Callback(hObject, eventdata, handles)
