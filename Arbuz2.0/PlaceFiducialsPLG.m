function varargout = PlaceFiducialsPLG(varargin)
% PLACEFIDUCIALSPLG MATLAB code for PlaceFiducialsPLG.fig
%      PLACEFIDUCIALSPLG, by itself, creates a new PLACEFIDUCIALSPLG or raises the existing
%      singleton*.
%
%      H = PLACEFIDUCIALSPLG returns the handle to a new PLACEFIDUCIALSPLG or the handle to
%      the existing singleton*.
%
%      PLACEFIDUCIALSPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLACEFIDUCIALSPLG.M with the given input arguments.
%
%      PLACEFIDUCIALSPLG('Property','Value',...) creates a new PLACEFIDUCIALSPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before PlaceFiducialsPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to PlaceFiducialsPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help PlaceFiducialsPLG

% Last Modified by GUIDE v2.5 23-Nov-2015 12:43:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PlaceFiducialsPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @PlaceFiducialsPLG_OutputFcn, ...
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


% --- Executes just before PlaceFiducialsPLG is made visible.
function PlaceFiducialsPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to PlaceFiducialsPLG (see VARARGIN)

% Choose default command line output for PlaceFiducialsPLG
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);
cameratoolbar;
% UIWAIT makes PlaceFiducialsPLG wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = PlaceFiducialsPLG_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

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

x = (1:256)*0.1;
y = (1:256)*0.1;
z = (1:39)*0.75;

box(handles.axes1,'on');
grid(handles.axes1,'on');

% x = (1:128)*0.1;
% y = (1:128)*0.1;
% z = (1:128)*0.1;


[X,Y,Z] = meshgrid(x,y,z);

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

