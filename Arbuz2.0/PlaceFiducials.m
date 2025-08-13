function varargout = PlaceFiducials(varargin)
% PLACEFIDUCIALS MATLAB code for PlaceFiducials.fig
%      PLACEFIDUCIALS, by itself, creates a new PLACEFIDUCIALS or raises the existing
%      singleton*.
%
%      H = PLACEFIDUCIALS returns the handle to a new PLACEFIDUCIALS or the handle to
%      the existing singleton*.
%
%      PLACEFIDUCIALS('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLACEFIDUCIALS.M with the given input arguments.
%
%      PLACEFIDUCIALS('Property','Value',...) creates a new PLACEFIDUCIALS or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PlaceFiducials_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PlaceFiducials_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help PlaceFiducials

% Last Modified by GUIDE v2.5 24-Nov-2015 11:16:58

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PlaceFiducials_OpeningFcn, ...
                   'gui_OutputFcn',  @PlaceFiducials_OutputFcn, ...
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


% --- Executes just before PlaceFiducials is made visible.
function PlaceFiducials_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to PlaceFiducials (see VARARGIN)

% Choose default command line output for PlaceFiducials

handles.output = hObject;
handles.isOK = false;
handles.A = eye(4);

% Update handles structure
guidata(hObject, handles);

cameratoolbar(handles.figure1);

if ~isempty(varargin)
  
  image = varargin{1};
  bwimage = image.Mask;
  
  A = image.Anative;
  handles.A = A;
  bbox = size(bwimage);
  
  Bounds1 = hmatrix_augment([1,1,1])*A;
  Bounds2 = hmatrix_augment(bbox)*A;
  handles.x = linspace(Bounds1(1),Bounds2(1),bbox(1));
  handles.y = linspace(Bounds1(2),Bounds2(2),bbox(2));
  handles.z = linspace(Bounds1(3),Bounds2(3),bbox(3));

  CC = bwconncomp(bwimage);
  nObjects = length(CC.PixelIdxList);
  CCsize = zeros(nObjects, 1);
  
  for ii=1:nObjects
    CCsize(ii,1) = length(CC.PixelIdxList{ii});
  end
  
  % remove those smaller that one slice
  str = {};
  handles.obj = {};
  kk = 1;
  for ii=1:nObjects
    if CCsize(ii) > 80
      handles.obj{end+1}.idx = CC.PixelIdxList{ii};
      handles.obj{end}.N   = length(handles.obj{end}.idx);
      str{end+1} = sprintf('fiducial %i', kk); kk = kk + 1;
    end
  end
  guidata(hObject, handles);
  
  set([handles.pmDelete, handles.pmJoin1, handles.pmJoin2], 'string', str, 'value', 1);
  
  Plot(handles);
  % UIWAIT makes PlaceFiducials wait for user response (see UIRESUME)
  uiwait(handles.figure1);
end

% --------------------------------------------------------------------
function varargout = PlaceFiducials_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.res;

delete(handles.figure1);

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)

arbuz_image = evalin('base', 'arbuz_image');
bwimage = arbuz_image.data;

% Find biggest areas

CC = bwconncomp(bwimage);
nObjects = length(CC.PixelIdxList); 
CCsize = zeros(nObjects, 1);

for ii=1:nObjects
  CCsize(ii,1) = length(CC.PixelIdxList{ii});
end

% remove those smaller that one slice
str = {};
handles.obj = {};
kk = 1;
for ii=1:nObjects
  if CCsize(ii) > 80
    handles.obj{end+1}.idx = CC.PixelIdxList{ii};
    handles.obj{end}.N   = length(handles.obj{end}.idx);
    str{end+1} = sprintf('fiducial %i', kk); kk = kk + 1;
  end
end
guidata(hObject, handles);

set([handles.pmDelete, handles.pmJoin1, handles.pmJoin2], 'string', str, 'value', 1);

Plot(handles);

% --------------------------------------------------------------------
function Plot(handles)

cla(handles.axes1);
hold(handles.axes1, 'on')
color = {'k', 'b', 'm', 'r', 'g', 'c', 'y', 'k', 'b', 'm', 'r', 'g', 'c'};
nObjects = length(handles.obj);

x = handles.x(:);
y = handles.y(:);
z = handles.z(:);

[X,Y,Z] = meshgrid(y,x,z);

for ii=1:nObjects
  % find best line through the object
  idx = handles.obj{ii}.idx;
  points = [X(idx),Y(idx),Z(idx)];
  N = size(points, 1);
  r0 = mean(points);
  xyz = points - repmat(r0, [N, 1]);
  [a,b,V]=svd(xyz,0);
  a = V(1:3,1)';

  handles.obj{ii}.r0 = r0;
  handles.obj{ii}.a = a;
  
  res = 0;
  for jj=1:handles.obj{ii}.N
    res = res + (norm(cross(points(jj,:) - r0, a)) / norm(a))^2;
  end
  handles.obj{ii}.std = sqrt(res/handles.obj{ii}.N);
  
  plot3(points(:,1), points(:,2), points(:,3), 'Color', color{ii}, 'Marker', '.', 'lineStyle', 'none', 'Parent', handles.axes1)
end

str = {};
for ii=1:nObjects
  r0 = handles.obj{ii}.r0;
  a = handles.obj{ii}.a;
  
  idx = handles.obj{ii}.idx;
  t = -10:0.1:10; 
  ft = repmat(r0,length(t),1) + repmat(a,length(t),1).*repmat(t',1,3);
  
  minx = min(X(idx)); maxx = max(X(idx));
  miny = min(Y(idx)); maxy = max(Y(idx));
  minz = min(Z(idx)); maxz = max(Z(idx));
  t = t(ft(:,1) >= minx & ft(:,1) <= maxx & ft(:,2) >= miny & ft(:,2) <= maxy & ft(:,3) >= minz & ft(:,3) <= maxz);
  t = t([1,end]);
  
  plot3(r0(1) + a(1)*t, r0(2) + a(2)*t, r0(3) + a(3)*t, '-', 'linewidth', 2, 'color', color{ii}, 'Parent', handles.axes1)
  str{ii} = sprintf('fiducial %i', ii);
end
axis(handles.axes1, 'equal')

legend(str)
box(handles.axes1,'on');
grid(handles.axes1,'on');

guidata(handles.figure1, handles);


% --------------------------------------------------------------------
function pbDelete_Callback(hObject, eventdata, handles)
FFF1 = get(handles.pmDelete, 'value');

handles.obj = handles.obj((1:length(handles.obj))~=FFF1);

nObjects = length(handles.obj);
guidata(hObject, handles);

str = {};
for ii=1:nObjects
    str{ii} = sprintf('fiducial %i', ii);
end

set([handles.pmDelete, handles.pmJoin1, handles.pmJoin2], 'string', str, 'value', 1);

Plot(handles);

% --------------------------------------------------------------------
function pushbutton3_Callback(hObject, eventdata, handles)
FFF1 = get(handles.pmJoin1, 'value');
FFF2 = get(handles.pmJoin2, 'value');

handles.obj{FFF1}.idx = [handles.obj{FFF1}.idx;handles.obj{FFF2}.idx];
handles.obj{FFF1}.N = length(handles.obj{FFF1}.idx);

handles.obj = handles.obj((1:length(handles.obj))~=FFF2);
nObjects = length(handles.obj);
guidata(hObject, handles);

str = {};
for ii=1:nObjects
    str{ii} = sprintf('fiducial %i', ii);
end

set([handles.pmDelete, handles.pmJoin1, handles.pmJoin2], 'string', str, 'value', 1);

Plot(handles);

% --------------------------------------------------------------------
function pbOK_Callback(hObject, eventdata, handles)
handles.isOK = true;

x = handles.x;
y = handles.y;
z = handles.z;

[X,Y,Z] = meshgrid(y,x,z);

handles.res = {};

nObjects = length(handles.obj);
for ii=1:nObjects
  r0 = handles.obj{ii}.r0;
  a = handles.obj{ii}.a;
  
  idx = handles.obj{ii}.idx;
  t = -10:0.1:10; 
  ft = repmat(r0,length(t),1) + repmat(a,length(t),1).*repmat(t',1,3);
  
  minx = min(X(idx)); maxx = max(X(idx));
  miny = min(Y(idx)); maxy = max(Y(idx));
  minz = min(Z(idx)); maxz = max(Z(idx));
  t = t(ft(:,1) >= minx & ft(:,1) <= maxx & ft(:,2) >= miny & ft(:,2) <= maxy & ft(:,3) >= minz & ft(:,3) <= maxz);
  t = t([1,end]);
  handles.res{ii}.n = ii;
  handles.res{ii}.A1 = r0 + t(1)*a; 
  handles.res{ii}.A2 = r0 + t(2)*a;
  
  handles.res{ii}.A1=handles.res{ii}.A1([2,1,3]);
  handles.res{ii}.A2=handles.res{ii}.A2([2,1,3]);
  
  handles.res{ii}.IJK1 =fix(hmatrix_augment(handles.res{ii}.A1)*inv(handles.A)+0.5);
  handles.res{ii}.IJK1 = handles.res{ii}.IJK1([1:3]);
  handles.res{ii}.IJK2 =fix(hmatrix_augment(handles.res{ii}.A2)*inv(handles.A)+0.5);
  handles.res{ii}.IJK2 = handles.res{ii}.IJK2([1:3]);
  
  handles.res{ii}.IJK1=handles.res{ii}.IJK1([2,1,3]);
  handles.res{ii}.IJK2=handles.res{ii}.IJK2([2,1,3]);  
end
guidata(handles.figure1, handles);

uiresume();


% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)
handles.isOK = false;
handles.res = {};
guidata(handles.figure1, handles);

uiresume();

