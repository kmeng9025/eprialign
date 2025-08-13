function varargout = SingleSliceRPT(varargin)
% SINGLESLICERPT M-file for SingleSliceRPT.fig
% Co-Registration GUI and plug-ins 

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2007

% Last Modified by GUIDE v2.5 13-Jun-2014 10:43:41

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @SingleSliceRPT_OpeningFcn, ...
                   'gui_OutputFcn',  @SingleSliceRPT_OutputFcn, ...
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
function SingleSliceRPT_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
handles.hh = varargin{1};
handles.cache = {};
handles.Rotate = 0;
% Update handles structure
guidata(hObject, handles);
pbUpdate_Callback(hObject, eventdata, handles);
UpdateGroupMenu(handles)

% --------------------------------------------------------------------
function varargout = SingleSliceRPT_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --------------------------------------------------------------------
function ForceRedraw(h)

handles = guidata(h);

if get(handles.cbRedraw, 'Value')
  pbShow_Callback([], [], handles);
end

% --------------------------------------------------------------------
function pbUpdate_Callback(hObject, eventdata, handles)
UpdateGroupMenu(handles)

% --------------------------------------------------------------------
function pbShow_Callback(hObject, eventdata, handles)
tic
% find which group we are showing
val = get(handles.pmShowGroup, 'Value');

if val == 1
  draw_list = arbuz_FindImage(handles.hh, 'v', '', '', {});
elseif val > 2
  if val-2 <= length(hhandles.Groups)
    draw_list = arbuz_FindImage(handles.hh, hhandles.Groups{val-2}.list, '', '', {});
  end
else draw_list = {};
end

% find all 2D pictures
find_list2D = arbuz_FindImage(handles.hh, draw_list, 'ImageType', '2D', {'Aslave', 'Ashow','data','FullName', 'SlaveList'});
if isempty(find_list2D)
  disp('SingleSlicesRPT: Nothing to show!');
  return;
end

val = get(handles.pmReferenceFrame, 'Value');
find_list = find_list2D(val);
Aproxy = find_list{1}.Aslave;

proxy_to_look = {};
if find_list{1}.SlaveIdx ~= -1
  tmp = arbuz_FindImage(handles.hh, {struct('ImageIdx', find_list{1}.ImageIdx, 'SlaveIdx', -1)},...
    'ImageType', '2D', {'Ashow','data','FullName', 'SlaveList'});
  for ii=1:length(tmp{1}.SlaveList)
    proxy_to_look{end+1}=tmp{1}.SlaveList{ii};
  end
else
  proxy_to_look = find_list{1}.SlaveList;  
end

fieldset = {'Ashow','data','Color','Name', 'Aslave'};
find_listProxyList = arbuz_FindImage(handles.hh, ...
  proxy_to_look, 'ImageType', 'CONTOUR', fieldset);
find_listProxyList1 = arbuz_FindImage(handles.hh, ...
  proxy_to_look, 'ImageType', 'XYZ', fieldset);
for ii=1:length(find_listProxyList1), find_listProxyList{end+1} = find_listProxyList1{ii}; end

find_LEG_contour = arbuz_FindImage(handles.hh, ...
  find_listProxyList, 'Name', 'LEG', fieldset);

pars = {'Ashow','data','FullName', 'SlaveList'};
find_list3D    = arbuz_FindImage(handles.hh, draw_list, 'ImageType', '3DEPRI', pars);
find_list3DPO2 = arbuz_FindImage(handles.hh, draw_list, 'ImageType', 'PO2_pEPRI', pars);
find_list3DAMP = arbuz_FindImage(handles.hh, draw_list, 'ImageType', 'AMP_pEPRI', pars);
find_list3DMRI = arbuz_FindImage(handles.hh, draw_list, 'ImageType', 'MRI', pars);
find_list3DDICOM = arbuz_FindImage(handles.hh, draw_list, 'ImageType', 'DICOM3D', pars);


find_AMP_EPRI = arbuz_FindImage(handles.hh, 'all', 'ImageType', 'AMP_pEPRI', {'Ashow','data','FullName'});

% select all 2D images
% find_list = find_list2D;
for ii=1:length(find_list3D), find_list{end+1} = find_list3D{ii}; end
for ii=1:length(find_list3DPO2), find_list{end+1} = find_list3DPO2{ii}; end
for ii=1:length(find_list3DAMP), find_list{end+1} = find_list3DAMP{ii}; end
for ii=1:length(find_list3DMRI), find_list{end+1} = find_list3DMRI{ii}; end
for ii=1:length(find_list3DDICOM), find_list{end+1} = find_list3DDICOM{ii}; end

if get(handles.cbUseMasterFrame, 'Value')
  r = htransform_vectors(find_list2D{val}.Ashow, [0,0,0]);
  A2frame = inv(hmatrix_translate([0,0,r(3)]));
else
  A2frame = inv(find_list2D{val}.Ashow);
end
A2frame = A2frame*hmatrix_rotate_z(handles.Rotate);

% find appropriate bounds
d_bounds = str2num(get(handles.eBounds, 'String'));
bounds = [Inf, -Inf, Inf, -Inf];
if get(handles.cbScale2Contour, 'value')
  for kk=1:length(find_listProxyList)
    switch find_listProxyList{kk}.ImageType
      case 'XYZ'
        %           arbuz_DrawPoint3D(ax, struct('data',plot_list{1}.ProxyList{kk}.data, 'Color', color, 'LineWidth', lw), ...
        %             plot_list{1}.ProxyList{kk}.Ashow * A2frame);
      case 'CONTOUR'
        bbox = contour2bbox(find_listProxyList{kk}.data, inv(Aproxy));
        bounds = [min([bounds(1), bbox(:,1)]), ...
          max([bounds(2), bbox(:,2)]), ...
          min([bounds(3), bbox(:,3)]), ...
          max([bounds(4), bbox(:,4)])];
        %       find_listProxyList{kk}.Proxy
    end
  end
end
if any(isinf(bounds))
  sz = size(find_list2D{val}.data);
  bounds = [1, sz(2), 1, sz(1)];
end

if ~isempty(d_bounds)
  dx = bounds(2)-bounds(1);
  dy = bounds(4)-bounds(3);
  bounds = bounds + [dx*d_bounds(1), -dx*d_bounds(2), ...
    dy*d_bounds(3), -dy*d_bounds(4)];
end

FigN = str2double(get(handles.eFigureN, 'String'));
SubPlotN = str2num(get(handles.eSubplot, 'String'));
if isempty(SubPlotN), SubPlotN = [1,1]; end

figure(FigN); 
set(FigN, 'Name', find_list2D{val}.FullName);
xmin = []; xmax = [];
ymin = []; ymax = [];
nColumns = length(find_list);
nRows    = SubPlotN(1);
theRow   = SubPlotN(2);
if (theRow == 1), clf; end
ax_coordinates = epr_CalcAxesPos(nRows, nColumns, [0.04 0.04], [0.02 0.02]);
ax_coordinates = ax_coordinates((theRow-1)*nColumns + [1:nColumns], :);
h = zeros(1, nColumns);

PrintLabels = get(handles.cbShowLabels, 'Value');

val_col = get(handles.pbColorContours, 'Value');
str_col = get(handles.pbColorContours, 'String');
proxy_color = str_col{val_col};

amp_treshold = 0.15;

for ii=1:length(find_list)
  h(ii) = axes('Position', ax_coordinates(ii,:)); hold on
  AA = find_list{ii}.Ashow * A2frame;
  find_list{ii}.AA = AA;

  % use cache rather than calculate
  use_cache = get(handles.cbUseCache, 'Value') && ...
    ii<=length(handles.cache) && ...
    is_nochange(find_list{ii}, handles.cache{ii});

  if use_cache
    x = handles.cache{ii}.x;
    y = handles.cache{ii}.y;
  end
      
  switch find_list{ii}.ImageType
    case '2D'
      if use_cache
        imagesc(x,y,handles.cache{ii}.im);
      else
        data = find_list{ii}.data;
        [im1, x, y] = project_image2D(data, AA);
        handles.cache{ii} = move_to_cache(find_list{ii}, x, y, im1, AA);
        hh = imagesc(x,y,im1);
        handles.cache{ii}.mask = [];
      end
      color = proxy_color;
    case '3DEPRI'
      if use_cache
        hh = imagesc(x,y,handles.cache{ii}.im, ...
          handles.cache{ii}.scale);
      else
        [im1, x, y] = project_image3D(find_list{ii}.data, AA, bounds);
        handles.cache{ii} = move_to_cache(find_list{ii}, x, y, im1, AA);
        handles.cache{ii}.scale = [0,max(find_list{ii}.data(:))*0.95];
        hh = imagesc(x,y,im1);
        handles.cache{ii}.mask = [];
      end
      color = 'w';
    case 'PO2_pEPRI'
      if use_cache
        hh = imagesc(x,y,handles.cache{ii}.im, ...
          handles.cache{ii}.scale);
      else
        [im1, x, y, AAA] = project_image3D(find_list{ii}.data, AA, bounds);
        handles.cache{ii} = move_to_cache(find_list{ii}, x, y, im1, AA);
        handles.cache{ii}.scale = [0,100];
        hh = imagesc(x,y,im1, [0,100]);
        handles.cache{ii}.mask = im1 ~= -1;

%         make treshold looking on the amplitude image
        if ~isempty(find_AMP_EPRI)
          % create mask within the leg
          mask = logical(project_contour2mask_2D(find_LEG_contour{1}.data, ...
            find_LEG_contour{1}.Ashow*A2frame*AAA, length(x), length(y)));
          im2 = project_image3D(find_AMP_EPRI{end}.data, find_list{ii}.Ashow * A2frame, bounds);
          max_int = max(im2(:));
          handles.cache{ii}.mask = (im2 > max_int *amp_treshold | mask)& ...
            handles.cache{ii}.mask;
        end

      end
      color = 'w';
      colormap(gca,jet);
    case 'AMP_pEPRI'
      if use_cache
        hh = imagesc(x,y,handles.cache{ii}.im, ...
          handles.cache{ii}.scale);
      else
        [im1, x, y] = project_image3D(find_list{ii}.data, AA, bounds);
        handles.cache{ii} = move_to_cache(find_list{ii}, x, y, im1, AA);
        handles.cache{ii}.scale = [0,max(find_list{ii}.data(:))*0.95];
        hh = imagesc(x,y,im1, [0,max(find_list{ii}.data(:))]);
        
        max_int = max(im1(:));
        handles.cache{ii}.mask = im1 > max_int *amp_treshold;
%         mask = [];
      end
      color = 'w';
      colormap(gca,jet);
    case {'MRI', 'DICOM3D'}
      if use_cache
        hh = imagesc(x,y,handles.cache{ii}.im, ...
          handles.cache{ii}.scale);
      else
        [im1, x, y] = project_image3D(find_list{ii}.data, AA, bounds);
        handles.cache{ii} = move_to_cache(find_list{ii}, x, y, im1, AA);
        handles.cache{ii}.scale = [0,max(find_list{ii}.data(:))*0.95];
        imagesc(x,y,im1, [0,max(find_list{ii}.data(:))*0.95]);
      end
      color = 'w';
      colormap(gca,bone);
      handles.cache{ii}.mask = [];
  end
  if use_cache && ~isempty(handles.cache{ii}.mask), set(hh,'AlphaData',handles.cache{ii}.mask); end
  
  xmin = min([xmin,x]); xmax = max([xmax,x]);
  ymin = min([ymin,y]); ymax = max([ymax,y]);

  ax = gca;
  % draw proxies
  for kk=1:length(find_listProxyList)
    switch find_listProxyList{kk}.ImageType
      case 'XYZ'
        find_listProxyList{kk}.Color.Color = color;
        find_listProxyList{kk}.Color.LineWidth = 1;
        find_listProxyList{kk}.Color.PrintLabel = PrintLabels;
        arbuz_DrawPoint2D(ax, find_listProxyList{kk}, ...
          find_listProxyList{kk}.Ashow * A2frame);
      case 'CONTOUR'
        find_listProxyList{kk}.Color.Color = color;
        find_listProxyList{kk}.Color.LineWidth = 2;
        find_listProxyList{kk}.Color.PrintLabel = PrintLabels;
        arbuz_DrawContour2D(ax, find_listProxyList{kk}, ...
          find_listProxyList{kk}.Ashow * A2frame);
    end
  end
 
  title(h(ii), find_list{ii}.FullName, 'Interpreter', 'none');
  axis image
end

if isempty(bounds), axis(h,[xmin xmax ymin ymax]);
else axis(h,bounds)
end

% set(h(:), 'XTick', [], 'YTick', [])
guidata(handles.figure1, handles);
disp('SingleSlicesRPT: Done.')
toc

% --------------------------------------------------------------------
function UpdateGroupMenu(handles)

str = {'Default';'No output'};

Groups = arbuz_get(handles.hh, 'Groups');
for ii=1:length(Groups)
    str{end+1} = Groups{ii}.Name;
end

val = max(get(handles.pmShowGroup, 'Value'), 1);
set(handles.pmShowGroup, 'String', str, 'Value', min(val, length(str)));
UpdateReferenceFrame(handles);

% --------------------------------------------------------------------
function pmShowGroup_Callback(hObject, eventdata, handles)
UpdateReferenceFrame(handles);

% --------------------------------------------------------------------
function UpdateReferenceFrame(handles)
% find which group we are showing
val = get(handles.pmShowGroup, 'Value');

if val == 1
  draw_list = arbuz_FindImage(handles.hh, 'v', '', '', {});
elseif val > 2
  if val-2 <= length(hhandles.Groups)
    draw_list = arbuz_FindImage(handles.hh, hhandles.Groups{val-2}.list, '', '', {});
  end
else draw_list = {};
end

% find all 2D pictures
find_list2D = arbuz_FindImage(handles.hh, draw_list, 'ImageType', '2D', {'Ashow','data','FullName'});

str = {};
for ii=1:length(find_list2D)
    str{end+1} = find_list2D{ii}.FullName;
end
if isempty(str), str = {'?'}; end

val = max(get(handles.pmReferenceFrame, 'Value'), 1);
set(handles.pmReferenceFrame, 'String', str, 'Value', min(val, length(str)));

% --------------------------------------------------------------------
function cache = move_to_cache(im, x, y, data, AA)

cache.x = x;
cache.y = y;
cache.im = data;
cache.AA = AA;
cache.ImageIdx = im.ImageIdx;
cache.SlaveIdx = im.SlaveIdx;

% --------------------------------------------------------------------
function ret = is_nochange(im1, im2)

ret = 1;
if isempty(im1) || isempty(im2) || ...
    im1.ImageIdx ~= im2.ImageIdx || im1.SlaveIdx ~= im2.SlaveIdx || ...
    ~isequal(im1.AA, im2.AA)
  ret = 0;
end

% --------------------------------------------------------------------
function slRotate_Callback(hObject, eventdata, handles)
val   = get(hObject,'Value'); % steps are 0.02
pos = handles.Rotate;

handles.Rotate = pos + val*50;
set(hObject,'Value',0);
set(handles.eRotate, 'String', num2str(handles.Rotate))
guidata(hObject, handles);
pbShow_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function eRotate_Callback(hObject, eventdata, handles)
handles.Rotate = str2num(get(handles.eRotate, 'String'));
set(handles.eRotate, 'String', num2str(handles.Rotate))
guidata(hObject, handles);
pbShow_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbBringUp_Callback(hObject, eventdata, handles)
FigN = str2double(get(handles.eFigureN, 'String'));
figure(FigN);


% --- Executes on button press in cbUseMasterFrame.
function cbUseMasterFrame_Callback(hObject, eventdata, handles)
% hObject    handle to cbUseMasterFrame (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cbUseMasterFrame


% --- Executes on selection change in pbColorContours.
function pbColorContours_Callback(hObject, eventdata, handles)
% hObject    handle to pbColorContours (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = get(hObject,'String') returns pbColorContours contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pbColorContours


% --- Executes during object creation, after setting all properties.
function pbColorContours_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pbColorContours (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in cbScale2Contour.
function cbScale2Contour_Callback(hObject, eventdata, handles)
% hObject    handle to cbScale2Contour (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cbScale2Contour


% --------------------------------------------------------------------
function pbInfo_Callback(hObject, eventdata, handles)
the_message = ['Plugin for visualization 3D images in the coordinate system of 2D image.\n',...
'1. Press [Update] button to load all visible 2D images\n',...
'2. Enter [Figure] number (f.e. 1) and [Subplot] (f.e. 2,1).\n',...
'  Subplot: first number: overall number of rows\n',...
'  Subplot: second number: row to plot. Figure is cleared when first row is plotted.\n',...
'3. Select 2D image from [Frame].\n',...
'4. Press [Show] button.\n\n', ...
'Hint: Many plugins can be used to generate one figure with many rows.'];
msgbox(sprintf(the_message),'Info', 'help')
