function varargout = CMeanPLG(varargin)
% CMEANPLG MATLAB code for CMeanPLG.fig
%      CMEANPLG, by itself, creates a new CMEANPLG or raises the existing
%      singleton*.
%
%      H = CMEANPLG returns the handle to a new CMEANPLG or the handle to
%      the existing singleton*.
%
%      CMEANPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CMEANPLG.M with the given input arguments.
%
%      CMEANPLG('Property','Value',...) creates a new CMEANPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before CMeanPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to CMeanPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help CMeanPLG

% Last Modified by GUIDE v2.5 03-Dec-2018 14:43:18

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @CMeanPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @CMeanPLG_OutputFcn, ...
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
function CMeanPLG_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};

plist = {'FullName'};
find_list3D = {};
find_list3DMRI = arbuz_FindImage(handles.hh, 'master', 'ImageType', 'MRI', plist);

for ii=1:length(find_list3DMRI), find_list3D{end+1} = find_list3DMRI{ii}; end

handles.find_list = find_list3D;
str = cell(1, length(find_list3D));
for ii=1:length(find_list3D)
  str{ii}=find_list3D{ii}.FullName;
end
set(handles.pmSource, 'String', str);
pos1 = get(handles.pmSource, 'Value');
fill_PopupMenu(handles, handles.find_list{pos1});

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = CMeanPLG_OutputFcn(hObject, eventdata, handles) 

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function pushbutton1_Callback(hObject, eventdata, handles)
image_name = handles.pmSource.String{handles.pmSource.Value};
mask_name = handles.pmSourceMask.String{handles.pmSourceMask.Value};
find_list = arbuz_FindImage(handles.hh, {image_name}, '', '', {'data'});
the_data = find_list{1}.data;
find_list = arbuz_FindImage(handles.hh, {image_name, mask_name}, '', '', {'data'});
the_mask = find_list{1}.data;

idx  = find(the_mask);
sz = size(the_data);
selection = reshape(the_data, [sz(1)*sz(2)*sz(3), sz(4)]);
selection = selection(idx, :);

% c-mean code
Nc = handles.popupmenu3.Value;
if Nc < 2, return; end
[centers,U] = fcm(selection,Nc);

figure;plot(centers'); 
for ii=1:Nc
  group = U(ii,:) > 0.5;
  new_image = create_new_image('','3DMASK',[]);
  new_image.data = zeros(sz(1:3)); 
  new_image.data(idx(group)) = 1;
  new_image.Name = sprintf('cluster_%i', ii);
  arbuz_AddImage(handles.hh, new_image, find_list{1}.Image);
end

new_image = create_new_image('','MRI',[]);
new_image.data = zeros(sz(1:3));
for ii=1:Nc
  group = U(ii,:) > 0.5;
  new_image.data(idx(group)) = ii+U(ii,group);
end
new_image.Name = sprintf('cmap');
arbuz_AddImage(handles.hh, new_image, find_list{1}.Image);
arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pmSource_Callback(hObject, eventdata, handles)
pos1 = get(handles.pmSource, 'Value');
str = get(handles.pmSource, 'String');
fill_PopupMenu(handles, str{pos1});

% --------------------------------------------------------------------
function fill_PopupMenu(handles, im_name)
ProxyList = arbuz_FindImage(handles.hh, {im_name}, '', '', {'SlaveList'});
find_mask3D = arbuz_FindImage(handles.hh, ProxyList{1}.SlaveList, ...
  'ImageType', '3DMASK', {'Name'});

if ~isempty(find_mask3D)
  str = cell(1, length(find_mask3D));
  for ii=1:length(find_mask3D)
    str{ii}=find_mask3D{ii}.Name;
  end
else
  str = cell(1, 1);
  str{1} = 'no mask is found';
end
set(handles.pmSourceMask, 'String', str, 'value', 1);

% --------------------------------------------------------------------
function new_image = create_new_image(name, type, data)
new_image = [];
new_image.ImageType = type;
new_image.A = eye(4);
new_image.isStore = 1;
new_image.isLoaded = 1;
new_image.Name = name;
new_image.data = data;

% --------------------------------------------------------------------
function pushbutton2_Callback(hObject, eventdata, handles)
image_name = handles.pmSource.String{handles.pmSource.Value};
mask_name = handles.pmSourceMask.String{handles.pmSourceMask.Value};
find_list = arbuz_FindImage(handles.hh, {image_name}, '', '', {'data'});
the_data = find_list{1}.data;
find_list = arbuz_FindImage(handles.hh, {image_name, mask_name}, '', '', {'data'});
the_mask = find_list{1}.data;

idx  = find(the_mask);
sz = size(the_data);
selection = reshape(the_data, [sz(1)*sz(2)*sz(3), sz(4)]);
selection = selection(idx, :);


f = @(x, t) x(1)*exp(-x(2)*t).*(1-exp(-x(3)*t))+x(4);

fp = 14;
t = -fp:size(selection, 2)-1-fp;
fit_idx = fp:size(selection, 2);

options = optimoptions('fminunc');
options.StepTolerance = 0.01;
options.Display = 'none';

figure(1);
x = zeros(size(selection, 1),4);
for ii=1:size(selection, 1)
%   figure(4);clf;
  x(ii,:) = lsqcurvefit(f,[7000, 1/1e3, 1/10, 6000],t(fit_idx),selection(ii,fit_idx));
%   plot(t(fit_idx), f(x(ii,:), t(fit_idx)), t, selection(ii,:))
%   pause(0.001)
end

x(x(:,3) < 0,3) = 0;
new_image = create_new_image('','MRI',[]);
new_image.data = zeros(sz(1:3));
new_image.data(idx) = x(:,3);
new_image.Name = sprintf('uptake');
arbuz_AddImage(handles.hh, new_image, find_list{1}.Image);
arbuz_UpdateInterface(handles.hh);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
