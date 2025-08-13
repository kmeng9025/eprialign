function varargout = Register3DPLG(varargin)
% REGISTER3DPLG MATLAB code for Register3DPLG.fig
%      REGISTER3DPLG, by itself, creates a new REGISTER3DPLG or raises the existing
%      singleton*.
%
%      H = REGISTER3DPLG returns the handle to a new REGISTER3DPLG or the handle to
%      the existing singleton*.
%
%      REGISTER3DPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in REGISTER3DPLG.M with the given input arguments.
%
%      REGISTER3DPLG('Property','Value',...) creates a new REGISTER3DPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Register3DPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Register3DPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Register3DPLG

% Last Modified by GUIDE v2.5 20-Nov-2015 13:52:12

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Register3DPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @Register3DPLG_OutputFcn, ...
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
function Register3DPLG_OpeningFcn(hObject, eventdata, handles, varargin) %#ok<*INUSL>
handles.hh = varargin{1};
handles.output = hObject;

handles.Ref1 = {};
handles.Ref2 = {};
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = Register3DPLG_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles) %#ok<*DEFNU>
im = arbuz_FindImage(handles.hh, 'master', '', '', {});

str = {};
for ii=1:length(im)
  str{end+1} = im{ii}.Image;
end
set(handles.pmImage1, 'string', str, 'value', 1);
set(handles.pmImage2, 'string', str, 'value', 1);

handles.Ref1 = {};
handles.Ref2 = {};
guidata(hObject, handles);
UpdateListbox(handles.lbImage1, handles.Ref1);
UpdateListbox(handles.lbImage2, handles.Ref2);


% --------------------------------------------------------------------
function pbAddImage1_Callback(hObject, eventdata, handles)

str = get(handles.pmImage1, 'string');
val = get(handles.pmImage1, 'val');

im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {'SlaveList'});
xyz = arbuz_FindImage(handles.hh, im{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Ashow', 'Tag1'});

str = {};
for ii=1:length(xyz)
  str{end+1} = xyz{ii}.Slave;
end

[sel, ok] = listdlg('ListString', str, 'SelectionMode', 'multiple', 'Name', 'Select anchors');

if ok
  for ii=1:length(sel)
    idx = sel(ii);    
    if isempty(xyz{idx}.Tag1)
      obj_ref = 1;
    else
      obj_ref = fix(str2double(xyz{idx}.Tag1));
    end
    arbuz_SetImage(handles.hh, xyz{idx}, 'Tag1', num2str(obj_ref));
    handles.Ref1{end+1} = struct('Name', str{idx}, ...
    'ImageIdx', xyz{idx}.ImageIdx, 'SlaveIdx', xyz{idx}.SlaveIdx, ...  
    'Data', xyz{idx}.data, 'Ashow', xyz{idx}.Ashow, 'Assign', obj_ref);
  end
  
  guidata(hObject, handles);
  UpdateListbox(handles.lbImage1, handles.Ref1);
end

% --------------------------------------------------------------------
function pbAddImage2_Callback(hObject, eventdata, handles)

str = get(handles.pmImage2, 'string');
val = get(handles.pmImage2, 'val');

im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {'SlaveList'});
xyz = arbuz_FindImage(handles.hh, im{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Ashow', 'Tag1'});

str = {};
for ii=1:length(xyz)
  str{end+1} = xyz{ii}.Slave;
end

[sel, ok] = listdlg('ListString', str, 'SelectionMode', 'multiple', 'Name', 'Select anchors');

if ok
  for ii=1:length(sel)
    idx = sel(ii);    
    if isempty(xyz{idx}.Tag1)
      obj_ref = 1;
    else
      obj_ref = fix(str2double(xyz{idx}.Tag1));
    end
    arbuz_SetImage(handles.hh, xyz{idx}, 'Tag1', num2str(obj_ref));
    handles.Ref2{end+1} = struct('Name', str{idx}, ...
      'ImageIdx', xyz{idx}.ImageIdx, 'SlaveIdx', xyz{idx}.SlaveIdx, ...
      'Data', xyz{idx}.data, 'Ashow', xyz{idx}.Ashow, 'Assign', obj_ref);
  end
  
  guidata(hObject, handles);
  UpdateListbox(handles.lbImage2, handles.Ref2);
end

% --------------------------------------------------------------------
function UpdateListbox(item_handle, data_to_update)

str = {};
for ii=1:length(data_to_update)
  str{end+1} = sprintf('%s (%i)', data_to_update{ii}.Name, data_to_update{ii}.Assign);
end

val = get(item_handle, 'value');
if isempty(val) || val == 0, val = 1; end
set(item_handle, 'string', str, 'value', min(val, length(str)))

% --------------------------------------------------------------------
function pbOptionsImage1_Callback(hObject, eventdata, handles)
val = get(handles.lbImage1, 'val');

res=inputdlg({'Assignment'}, ['Anchor ', handles.Ref1{val}.Name], 1, {num2str(handles.Ref1{val}.Assign)});

if ~isempty(res)
  handles.Ref1{val}.Assign = fix(str2double(res{1}));
  arbuz_SetImage(handles.hh, {handles.Ref1{val}.ImageIdx, handles.Ref1{val}.SlaveIdx}, 'Tag1', res{1});
  guidata(hObject, handles);  
  UpdateListbox(handles.lbImage1, handles.Ref1);
end

% --------------------------------------------------------------------
function pbOptionsImage2_Callback(hObject, eventdata, handles)
val = get(handles.lbImage2, 'val');

res=inputdlg({'Assignment'}, ['Anchor ', handles.Ref2{val}.Name], 1, {num2str(handles.Ref2{val}.Assign)});

if ~isempty(res)
  handles.Ref2{val}.Assign = fix(str2double(res{1}));
  arbuz_SetImage(handles.hh, {handles.Ref2{val}.ImageIdx, handles.Ref2{val}.SlaveIdx}, 'Tag1', res{1});
  guidata(hObject, handles);  
  UpdateListbox(handles.lbImage2, handles.Ref2);
end

% --------------------------------------------------------------------
function pbBroadSearch_Callback(hObject, eventdata, handles)
str = get(handles.pmImage2, 'string');
val = get(handles.pmImage2, 'val');
im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {'Aprime'});

AA = transform_1(handles);

arbuz_SetImage(handles.hh, im, 'Aprime', im{1}.Aprime * AA);

arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function pbTransform_Callback(hObject, eventdata, handles) 
str = get(handles.pmImage2, 'string');
val = get(handles.pmImage2, 'val');
im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {'Aprime'});

AA = transform_2(handles);

arbuz_SetImage(handles.hh, im, 'Aprime', im{1}.Aprime * AA);

arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function pbClearTransform_Callback(hObject, eventdata, handles) 
str = get(handles.pmImage2, 'string');
val = get(handles.pmImage2, 'val');
im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {});

arbuz_SetImage(handles.hh, im, 'Aprime', eye(4));
arbuz_RedrawAll(handles.hh);

% --------------------------------------------------------------------
function pbDone_Callback(hObject, eventdata, handles) 
closereq;

% --------------------------------------------------------------------
function pmImage1_Callback(hObject, eventdata, handles) 
handles.Ref1 = {};
guidata(hObject, handles);
UpdateListbox(handles.lbImage1, handles.Ref1);

% --------------------------------------------------------------------
function pmImage2_Callback(hObject, eventdata, handles) 
handles.Ref2 = {};
guidata(hObject, handles);
UpdateListbox(handles.lbImage2, handles.Ref2);

% --------------------------------------------------------------------
function lbImage1_Callback(hObject, eventdata, handles) 
persistent chk1
if isempty(chk1)
      chk1 = 1;
      pause(0.5); %Add a delay to distinguish single click from a double click
      if chk1 == 1
%           fprintf(1,'\nI am doing a single-click.\n\n');
          chk1 = [];
      end
else
      chk1 = [];
      pbOptionsImage1_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function lbImage2_Callback(hObject, eventdata, handles) 
persistent chk2
if isempty(chk2)
      chk2 = 1;
      pause(0.5); %Add a delay to distinguish single click from a double click
      if chk2 == 1
%           fprintf(1,'\nI am doing a single-click.\n\n');
          chk2 = [];
      end
else
      chk2 = [];
      pbOptionsImage2_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function pbRemoveImage1_Callback(hObject, eventdata, handles) %#ok<*INUSD>
arbuz_ShowMessage(handles.hh, 'Not implemented. Use image selection to clear anchors.');

% --------------------------------------------------------------------
function pbRemoveImage2_Callback(hObject, eventdata, handles)
arbuz_ShowMessage(handles.hh, 'Not implemented. Use image selection to clear anchors.');

% --------------------------------------------------------------------
function pbFix_Callback(hObject, eventdata, handles)
% str = get(handles.pmImage2, 'string');
% val = get(handles.pmImage2, 'val');
% im = arbuz_FindImage(handles.hh, {struct('Image', str{val})}, '', '', {'SlaveList'});

arbuz_ApplyTransformation(handles.hh, '', 'FIX');

% --------------------------------------------------------------------
% distance(x0,y0,z0,a,b,c) = sum{[c(yi - y0) - b(z - z0)]^2 +
%                           [a(zi - z0) - c(x - x0)]^2 +
%                           [b(xi - x0) - a(y - y0)]^2}

% --------------------------------------------------------------------
% points is a Nx3 array where N is the number of points
function [r0, a] = get_best_line3D(points)

N = size(points, 1);
r0 = mean(points);
xyz = points - repmat(r0, [N, 1]);
[a,b,V]=svd(xyz,0); %#ok<ASGLU>
a = V(1:3,1)';   

% --------------------------------------------------------------------
function fiducials = get_fiducials(anchors)

% find fiducials
fiducials = [];
for ii=1:10
  points = [];
  for jj=1:length(anchors)
    if anchors{jj}.Assign == ii
      points(end+1, :) = htransform_vectors(anchors{jj}.Ashow, anchors{jj}.Data);
    end
  end
  if ~isempty(points)
    [r0, a] = get_best_line3D(points);
    fiducials(end+1, :) = [r0,a];
  end
end

% --------------------------------------------------------------------
function sorted_anchors = get_anchors(anchors)
sorted_anchors = [];
for ii=1:10
  for jj=1:length(anchors)
    if anchors{jj}.Assign == ii
      sorted_anchors(end+1, :) = [htransform_vectors(anchors{jj}.Ashow, anchors{jj}.Data), ii];
    end
  end
end

% --------------------------------------------------------------------
function transform_list = update_transformations(handles, transform_list)

for ii=1:length(transform_list)
  xyz = arbuz_FindImage(handles.hh, {struct('ImageIdx', transform_list{ii}.ImageIdx, 'SlaveIdx', transform_list{ii}.SlaveIdx)}, ...
    '', '', {'data', 'ACURRENT', 'Anext'});
 transform_list{ii}.data = xyz{1}.data;
 transform_list{ii}.Ashow = xyz{1}.Acurrent;
 transform_list{ii}.Anext = xyz{1}.Anext; 
end

% --------------------------------------------------------------------
function A = transform_1(handles)

handles.Ref1 = update_transformations(handles, handles.Ref1);
handles.Ref2 = update_transformations(handles, handles.Ref2);

% REF image anchors
anchorsREF = get_anchors(handles.Ref1);

% REF image fiducials
fiducials = get_fiducials(handles.Ref1);

% DEPENDENT image anchors
anchors = get_anchors(handles.Ref2);

% here Pxyz any point and RSxyz = [R0, A0] - line definition, R0 any
% point, A0 parameteric slope constants xyz = R0 + A0*t
% distance = @(Pxyz, RSxyz) norm(cross(Pxyz - RSxyz(1:3), RSxyz(4:6))) / norm(RSxyz(4:6));

all_tags = unique(anchors(:,4));
P = perms(all_tags);

for perm_idx = 1:size(P, 1)
  idxANR = P(perm_idx, :);
  
  % run a search for various major rotations
  for jj=1:4
    switch jj
      case 1, A0 = eye(4);
      case 2, A0 = hmatrix_rotate_x(180);
      case 3, A0 = hmatrix_rotate_y(180);
      case 4, A0 = hmatrix_rotate_z(180);
    end
    
    % form a fitting array
    fit_array = htransform_vectors(A0, anchors(:,1:3));
    for ii = 1:size(anchors, 1)
      anchor_tag = anchors(ii,4);
      fit_array(ii, 4:9) = fiducials(idxANR(anchor_tag),:);
    end
    
    options = optimset('TolFun',1e-4);
    RESALL(perm_idx, jj,:) = fminsearch(@fit_func1, [0,0,0,0,0,0], options, fit_array);
    ERR(perm_idx, jj) = fit_func1(squeeze(RESALL(perm_idx, jj,:))', fit_array);
  end
end

[minERR, minIDXa] = min(ERR);
[minERR, minIDX2] = min(minERR);
minIDX1 = minIDXa(minIDX2);
fprintf('Optimization error: %4.3f\n', minERR);

switch minIDX2
  case 1, A0 = eye(4);
  case 2, A0 = hmatrix_rotate_x(180);
  case 3, A0 = hmatrix_rotate_y(180);
  case 4, A0 = hmatrix_rotate_z(180);
end
RES = squeeze(RESALL(minIDX1, minIDX2, :))';

A1 = hmatrix_rotate_euler(RES(4:6)) * hmatrix_translate(RES(1:3));

A = A0*A1;

% Update anchors assignment
new_tags = P(minIDX1,:);
for ii = 1:length(handles.Ref2)
    new_tag = new_tags(handles.Ref2{ii}.Assign);
    handles.Ref2{ii}.Assign = new_tag;
    arbuz_SetImage(handles.hh, {handles.Ref2{ii}.ImageIdx, handles.Ref2{ii}.SlaveIdx}, 'Tag1', num2str(new_tag));
end

guidata(handles.figure1, handles);
UpdateListbox(handles.lbImage2, handles.Ref2);


% x = fminsearch(@fit_func2, [0,0,0], options, fit_array);
% disp(x)
% disp(sprintf('Fit error: %5.3f', fit_func2(x, fit_array)));
% A = hmatrix_translate(x(1:3));

% x = fminsearch(@fit_func3, [0,0,0], options, fit_array);
% % disp(x)
% disp(sprintf('Fit error: %5.3f', fit_func3(x, fit_array)));
% A = hmatrix_rotate_euler(x(1:3));

% colors = {'r','b','m'};
% figure(1); hold on;
% for ii=1:max(anchorsREF(:, 4))
%   idx = anchorsREF(:, 4) == ii;
%   plot3(anchorsREF(idx, 1), anchorsREF(idx, 2), anchorsREF(idx, 3), ['.-', colors{ii}]); hold on;
% end


% anchors_transformed= htransform_vectors(A, anchors(:,1:3));
% for ii=1:max(anchors(:, 4))
%   idx = anchors(:, 4) == ii;
%   plot3(anchors_transformed(idx, 1), anchors_transformed(idx, 2), anchors_transformed(idx, 3), ['p-', colors{ii}]); hold on;  
% end
% for ii=1:size(fiducials, 1)
%   p = fiducials(1:3);
% end
% grid on

% --------------------------------------------------------------------
function A = transform_2(handles)

handles.Ref1 = update_transformations(handles, handles.Ref1);
handles.Ref2 = update_transformations(handles, handles.Ref2);

% REF image anchors
% anchorsREF = get_anchors(handles.Ref1);

% REF image fiducials
fiducials = get_fiducials(handles.Ref1);

% DEPENDENT image anchors
anchors = get_anchors(handles.Ref2);

% form a fitting array
fit_array = anchors(:,1:3);
for ii = 1:size(anchors, 1)
  fit_array(ii, 4:9) = fiducials(anchors(ii,4),:);
end

options = optimset('TolFun',1e-4);
RES = fminsearch(@fit_func1, [0,0,0,0,0,0], options, fit_array);
minERR = fit_func1(RES, fit_array);

fprintf('Optimization error: %4.3f\n', minERR);

A = hmatrix_rotate_euler(RES(4:6)) * hmatrix_translate(RES(1:3));

% --------------------------------------------------------------------
function error = fit_func1(x, fit_array)
A = hmatrix_rotate_euler(x(4:6)) * hmatrix_translate(x(1:3));
dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(A, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
  point_to_line_distance(dist_array(ii,:));
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function error = fit_func2(x, fit_array)
A = hmatrix_translate(x(1:3));
dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(A, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function error = fit_func3(x, fit_array)
A = hmatrix_rotate_euler(x(1:3));
dist_array = fit_array;
dist_array(:,1:3) = htransform_vectors(A, fit_array(:,1:3));
error = 0;
for ii = 1:size(fit_array, 1)
  error = error + point_to_line_distance(dist_array(ii,:)).^2;
end
error = sqrt(error);
% disp(error);

% --------------------------------------------------------------------
function res = point_to_line_distance(FA) 

res = norm(cross(FA(1:3) - FA(4:6), FA(7:9))) / norm(FA(7:9));

% --------------------------------------------------------------------
function pbInfo_Callback(hObject, eventdata, handles)
the_message = ['Instructions:\n',...
'1. Press [Update] button\n',...
'2. Select static and rotating images\n',...
'3. Add anchors for registration using [+] button\n',...
'4. [Auto Tag] or manually assign anchors to fiducials by number (1,2,3 etc) using [?] button or double-click.\n',...
'5. Press [Register] or [Broad Search]. [Broad search] is ~10x slower but works better far from the solution.\n',...
'6. Repeat [Register] few times for stable solution.\n', ...
'7. [Fix] to fix results.\n\n',...
'Note: This plugin uses Tag1 field for object identification.'];
msgbox(sprintf(the_message),'Info', 'help')

% --------------------------------------------------------------------
function pbTag_Callback(hObject, eventdata, handles)
for ii = 1:length(handles.Ref1)
  a = regexp(handles.Ref1{ii}.Name, 'XYZ(?<tag>\d*)', 'names');
  if ~isempty(a) && isstruct(a)
    new_tag = fix(str2double(a.tag));
    handles.Ref1{ii}.Assign = new_tag;
    arbuz_SetImage(handles.hh, {handles.Ref1{ii}.ImageIdx, handles.Ref1{ii}.SlaveIdx}, 'Tag1', num2str(new_tag));
  end
end

for ii = 1:length(handles.Ref2)
  a = regexp(handles.Ref2{ii}.Name, 'XYZ(?<tag>\d*)', 'names');
  if ~isempty(a) && isstruct(a)
    new_tag = fix(str2double(a.tag));
    handles.Ref2{ii}.Assign = new_tag;
    arbuz_SetImage(handles.hh, {handles.Ref2{ii}.ImageIdx, handles.Ref2{ii}.SlaveIdx}, 'Tag1', num2str(new_tag));
  end
end

guidata(hObject, handles);
UpdateListbox(handles.lbImage1, handles.Ref1);
UpdateListbox(handles.lbImage2, handles.Ref2);

% --------------------------------------------------------------------
function pbCurrentError_Callback(hObject, eventdata, handles)

handles.Ref1 = update_transformations(handles, handles.Ref1);
handles.Ref2 = update_transformations(handles, handles.Ref2);

% REF image anchors
% anchorsREF = get_anchors(handles.Ref1);

% REF image fiducials
fiducials = get_fiducials(handles.Ref1);

% DEPENDENT image anchors
anchors = get_anchors(handles.Ref2);

% form a fitting array
fit_array = anchors(:,1:3);
for ii = 1:size(anchors, 1)
  fit_array(ii, 4:9) = fiducials(anchors(ii,4),:);
end

plot_result(fit_array);

total_error = 0;
for ii = 1:size(fit_array, 1)
  error = point_to_line_distance(fit_array(ii,:));
  total_error = total_error + error.^2;
  fprintf('Deviation of point %d: %4.3f\n', ii, error);
end
total_error = sqrt(total_error);

fprintf('Optimization error: %4.3f\n', total_error);

% --------------------------------------------------------------------
function plot_result(fit_array)
colors = {'r','b','m'};
figure(5); clf; hold on; grid on; axis equal

groups{1} = 1:2;
groups{2} = 3:4;
groups{3} = 5:6;

anchors = fit_array(:,1:3);
minx = min(anchors(:,1))-1; maxx = max(anchors(:,1))+1;
miny = min(anchors(:,2))-1; maxy = max(anchors(:,2))+1;
minz = min(anchors(:,3))-1; maxz = max(anchors(:,3))+1;

t = -20:0.1:20;

str = {};
for ii=1:length(groups)
  str{end+1} = sprintf('%i', ii);
  plot3(anchors(groups{ii},1),anchors(groups{ii},2),anchors(groups{ii},3),'o-', 'Color', colors{ii});
  
  r0 = fit_array(groups{ii}(1),4:6);
  a = fit_array(groups{ii}(1),7:9);
   
  ft = repmat(r0,length(t),1) + repmat(a,length(t),1).*repmat(t',1,3);

  %  & ft(:,2) >= miny & ft(:,2) <= maxy & ft(:,3) >= minz & ft(:,3) <= maxz
  t1 = t(ft(:,1) >= minx & ft(:,1) <= maxx & ft(:,2) >= miny & ft(:,2) <= maxy & ft(:,3) >= minz & ft(:,3) <= maxz);
  if ~isempty(t1)
    t1 = t1([1,end]);
    str{end+1} = sprintf('%i', ii);
    plot3(r0(1) + a(1)*t1, r0(2) + a(2)*t1, r0(3) + a(3)*t1, ':', 'Color', colors{ii}, 'linewidth', 3);
  end
  t1 = t(ft(:,1) >= minx-2 & ft(:,1) <= maxx+2 & ft(:,2) >= miny-2 & ft(:,2) <= maxy+2 & ft(:,3) >= minz-2 & ft(:,3) <= maxz+2);
  if ~isempty(t1)
    t1 = t1([1,end]);
    str{end+1} = sprintf('%i', ii);
    plot3(r0(1) + a(1)*t1, r0(2) + a(2)*t1, r0(3) + a(3)*t1, ':', 'Color', colors{ii}, 'linewidth', 1);
  end
end

axis equal
% camproj(5,'orthographic')
legend(str);
view(45, 45);


% 7.783136 -7.403512 5.971846
%
%
%
