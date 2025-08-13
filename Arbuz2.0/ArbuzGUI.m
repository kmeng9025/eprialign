function varargout = ArbuzGUI(varargin)
% Co-Registration GUI and plug-ins

% University of Chicago Medical Center
% Department of Radiation and Cellular Oncology
% B. Epel    2007-2008

% Project started 27-Nov-2007

% Last Modified by GUIDE v2.5 21-Apr-2023 11:24:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @ArbuzGUI_OpeningFcn, ...
  'gui_OutputFcn',  @ArbuzGUI_OutputFcn, ...
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
function ArbuzGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
handles.output = hObject;

% Assign database pointer (the same as GUI)
% This is to separate GUI and database pointer usage
handles.hGUI = handles.MainFigure; 

inifilename = epr_GetIniPath('ArbuzGUI');
handles.ini = inimanage(inifilename);
handles.ini.Directories = safeget(handles.ini, 'Directories', []);

handles.ico = load('ArbuzICONS');

arbuz_InitializeProject(handles.hGUI);

handles.RecentProjectList = {};
handles.saves = {};

handles.Constants.Group_3D         = -3;
handles.Constants.Group_2D         = -2;
handles.Constants.Folder           = -1;
    
handles.Constants.Group            = -10;
handles.Constants.Coordinate       = -20;
handles.Constants.Sequence         = -30;
handles.Constants.Transformation   = -40;
handles.Constants.Action           = -50;

handles.ImageTypes = {...
  '2D',...
  '3DEPRI',...
  'CONTOUR',...
  'XYZ',...
  '3DSURFACE',...
  'pEPROI',...
  'PO2_pEPRI',...
  'AMP_pEPRI',...
  'MRI',...
  '3DMASK',...
  'RAW',...
  'AMIRA3D',...
  'DICOM3D',...
  'SHAPE3D',...
  'GENERIC',...
  'FITRESULT', ...
  'IDL',...
  'WORKSPACE',...
  'MAT-GENERAL',...
  'BIN',...
  'CEST_ML'};

% Data filetypes
handles.FileTypes = {...
  '2dseq;*.am;*.dcm;*.img;*.jpg;*.mat;*.tiff;*.tif;*.dat','All supported extensions';...
  '*.am', 'AMIRA image files (*.am)';...
  '2dseq;*.img', 'MRI image files (2dseq;*.img)'; ...
  '*.img', 'OXY image files (*.img)'; ...
  '*.jpg', 'JPEG image files (*.jpg)';...
  '*.d01', 'SpecMan files (*.d01)'; ...
  '*.dcm', 'DICOM files (*.dcm)'; ...
  '*.m', 'Matlab scripts (*.m)'; ...
  '*.mat', 'Matlab files (*.mat)'; ...
  '*.raw', 'Raw data (*.raw)'; ...
  '*.tiff;*.tif', 'TIFF image files (*.tiff,*.tif)';...
  '*', 'IDL files (*.*)';...
  '*.dat', 'Variosu data files (*.dat)';...
  '*.*', 'All files (*.*)'};

% Data types
set(handles.pmImageType, 'String', handles.ImageTypes);

set([handles.lbObjects, handles.uipanelToolbar, handles.uipanel1, handles.uipanel2, handles.pmRegStage, ...
    handles.uipanel5, handles.eLogWindow], 'units', 'pixels');

icon_shift = 0.1;
handles.TB_buttons{1} = struct('icon', 'DefaultViewer','Callback','ArbuzGUI(''pbDefaultViewer_Callback'', guidata(gcbo), ''SliceMaskEdit'')','tips','Default Viewer');
handles.TB_buttons{2} = struct('icon', 'Information','Callback','ArbuzGUI(''pbImageInfo_Callback'',gcbo,[],guidata(gcbo))', 'tips','Information');
handles.TB_buttons{3} = struct('icon', 'SliceMaskEdit','Callback','ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''SliceMaskEdit'')', 'tips','');
handles.TB_buttons{4} = struct('icon', 'RotateImage','Callback','ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''RotateImage'')', 'tips','');
handles.TB_buttons{5} = struct('icon', 'CreateCoordinates','Callback','ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''CreateCoordinates'')', 'tips','');
handles.TB_buttons{6} = struct('icon', 'MaskTransfer','Callback','ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''ImageTransform'')', 'tips','');
handles.TB_buttons{7} = struct('icon', 'CropAndResample','Callback','ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''CropAndResample'')', 'tips','');
handles.TB_buttons{8} = struct('icon', 'NotesEdit','Callback','ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''EditNotes'')', 'tips','');
handles.ToolBarGray = [1,2,3,7];
for ii=1:length(handles.TB_buttons)
  icon = handles.ico.(handles.TB_buttons{ii}.icon);
  handles.ToolBar(ii) = uicontrol('Style', 'pushbutton', 'Parent', handles.uipanelToolbar, ...
    'Units', 'pixels', 'Position', [2+(ii+icon_shift-1)*(52+2), 2, 52, 52], ...
    'Callback',handles.TB_buttons{ii}.Callback,'CData', icon,'TooltipString',handles.TB_buttons{ii}.icon,...
    'TooltipString',handles.TB_buttons{ii}.tips);
end

% load plugins
DefaultPath = fileparts(which('ArbuzGUI'));
handles.ini.Directories.PluginsPath = DefaultPath;
% handles.ini.Directories.PluginsPath = safeget(handles.ini.Directories, 'PluginsPath', DefaultPath);
if ~strcmp(DefaultPath, handles.ini.Directories.PluginsPath)
  if ~exist(handles.ini.Directories.PluginsPath, 'dir')
    handles.ini.Directories.PluginsPath = DefaultPath;
  end
end

% remove old list
h = get(handles.mFile, 'Children');
for ii=1:length(h), if strfind(h(ii).Tag, 'RecentProject'), delete(h(ii)); end; end

% load recent files
RecentProjectList = safeget(handles.ini, 'RecentProjectList', []);
if ~isempty(RecentProjectList)
  FLDs = fieldnames(RecentProjectList);
  for ii=1:length(FLDs), handles.RecentProjectList{end+1}.fname = getfield(RecentProjectList, FLDs{ii}); end
  for ii = 1:length(handles.RecentProjectList)
    handles.RecentProjectList{ii}.hmenu = uimenu(handles.mFile, ...
      'Label', epr_ShortFileName(handles.RecentProjectList{ii}.fname, 30), ...
      'Callback', 'ArbuzGUI(''OpenProject_Callback'',gcbo,guidata(gcbo))', ...
      'Tag', 'RecentProject', 'UserData', ii);
    if ii == 1, set(handles.RecentProjectList{ii}.hmenu, 'Separator', 'on'); end
  end
end

guidata(handles.hGUI, handles);
mPluginReloadAction_Callback(hObject, eventdata, guidata(hObject));
mReportsReloadReports_Callback(hObject, eventdata, guidata(hObject));

% Update object list index
UpdateListbox(0,guidata(hObject));
lbObjects_Callback(handles.lbObjects, [], guidata(hObject));
AddMessage(handles, 'ArbuzGUI 2.0');

% Load project from the SECOND argument of the function
if nargin > 4
  OpenProject(varargin{2}, guidata(hObject));

  handles = guidata(hObject);
  handles.ini.Directories.ProjectPath = fileparts(varargin{2});

  AddRecentProjectList(handles, varargin{2});
end

% --------------------------------------------------------------------
function varargout = ArbuzGUI_OutputFcn(hObject, eventdata, handles) %#ok<*INUSL>
varargout{1} = handles.output;

% --------------------------------------------------------------------
function MainFigure_DeleteFcn(hObject, eventdata, handles) 
if isfield(handles, 'ini')
  RecentProjectList = [];
  for ii=1:length(handles.RecentProjectList)
    RecentProjectList = setfield(RecentProjectList, ['F', num2str(ii)], handles.RecentProjectList{ii}.fname);
  end
  handles.ini.RecentProjectList = RecentProjectList;
  inimanage(epr_GetIniPath('ArbuzGUI'), handles.ini);
end

% --------------------------------------------------------------------
function MainFigure_ResizeFcn(hObject, eventdata, handles) 
fig_size = get(handles.MainFigure, 'Position');
fig_size(3) = max(fig_size(3), 640);
fig_size(4) = max(fig_size(4), 420);
set(handles.MainFigure, 'Position', fig_size);

lbObjectsMenuH = 26;
lbObjectsMenuW = fig_size(3)-474;

set(handles.panImageSelector, 'Position', [1,fig_size(4)-lbObjectsMenuH,lbObjectsMenuW,lbObjectsMenuH]);
set(handles.lbObjects, 'Position', [1,1,lbObjectsMenuW-2,fig_size(4)-lbObjectsMenuH]);

set(handles.uipanelToolbar, 'Position', [lbObjectsMenuW,fig_size(4)-54,fig_size(3)-lbObjectsMenuW-2,54]);
set(handles.uipanel1, 'Position', [lbObjectsMenuW,fig_size(4)-54-150,fig_size(3)-lbObjectsMenuW-2,150]);
set(handles.uipanel5, 'Position', [lbObjectsMenuW,fig_size(4)-54-150-80,150,80]);
set(handles.uipanel2, 'Position', [lbObjectsMenuW+150,fig_size(4)-54-150-140,fig_size(3)-lbObjectsMenuW-2-150,140]);
set(handles.eLogWindow, 'Position', [lbObjectsMenuW,3,fig_size(3)-lbObjectsMenuW-2,fig_size(4)-350]);
set(handles.pmRegStage, 'Position', [lbObjectsMenuW,fig_size(4)-50-290+22,150,22]);
set(handles.pmSeeStage, 'Position', [lbObjectsMenuW,fig_size(4)-50-290,150,22]);

% --------------------------------------------------------------------
function set_highlighted(handles)
pos = get(handles.lbObjects, 'Value');
highlighted = [];

for ii=1:length(pos)
  idx = handles.ListBoxIndex{pos(ii)};
  if idx.ImageIndex > 0
    highlighted(end+1, :) = [idx.ImageIndex, idx.ProxyIndex];
  end
end

arbuz_set(handles.hGUI, 'Highlighted', highlighted);

% --------------------------------------------------------------------
function lbObjects_Callback(hObject, eventdata, handles) 

set_highlighted(handles);

pos = get(handles.lbObjects, 'Value');
if isempty(pos), return;
elseif length(pos) > 1
  return;
end

comments = '';
selection_name = '>';
enable_viewer = 'on';
ctrl_imtype = 'off';
ctrl_set_name = 'on';
ctrl_reload = 'off';
ctrl_filename = 'off';
ctrl_addmaster = 'off';
ctrl_addslave = 'off';
ctrl_browser = 'off';
idx = handles.ListBoxIndex{pos};
% This is an image
if idx.ImageIndex > 0
  im = arbuz_FindImage(handles.hGUI, [idx.ImageIndex,idx.ProxyIndex], '', '', ...
    {'Name', 'FileName', 'isStore', 'Color'});
  ctrl_reload = 'on';
  selection_name = im{1}.Name;
  set(handles.eFileName, 'String', im{1}.FileName);
  set(handles.cbVisible, 'Value', im{1}.Visible);
  set(handles.cbStoreInProject, 'Value', im{1}.isStore)
  if idx.ProxyIndex == -1, ctrl_addslave = 'on'; end
  for ii=1:length(handles.ImageTypes)
    if strcmp(im{1}.ImageType, handles.ImageTypes{ii})
       set(handles.pmImageType, 'Value', ii); break;
    end
    switch im{1}.ImageType
      case 'XYZ', set(handles.cbVisible, 'ForegroundColor', safeget(im{1}.Color, 'Color',[0 0 0]), 'FontWeight', 'bold');
      case '3DSURFACE', set(handles.cbVisible, 'ForegroundColor', safeget(im{1}.Color, 'EdgeColor',[0 0 0]), 'FontWeight', 'bold');
      otherwise, set(handles.cbVisible, 'ForegroundColor', [0,0,0]);
    end
  end
  ctrl_filename = 'on';
else
  switch idx.ImageIndex
    case handles.Constants.Folder 
      ctrl_set_name = 'off'; enable_viewer = 'off'; comments = 'This is a folder.';
    case handles.Constants.Group_2D
      ctrl_set_name = 'off'; enable_viewer = 'off'; ctrl_imtype = 'on'; ctrl_filename = 'on'; ctrl_addmaster = 'on';
      ctrl_browser = 'on';
      comments = '[Browse] for an image. Edit [name], choose [type]. Press [Add].';
    case handles.Constants.Group_3D 
      ctrl_browser = 'on';
      ctrl_set_name = 'off'; enable_viewer = 'off'; ctrl_imtype = 'on';  ctrl_filename = 'on'; ctrl_addmaster = 'on';
      comments = '[Browse] for an image. Edit [name], choose [type]. Press [Add].';
    case handles.Constants.Group, comments = 'This is a stored visualization group.';
    case handles.Constants.Coordinate, comments = 'This is a stored coordinate system.';
    case handles.Constants.Sequence, comments = 'This is a sequence of transformations.';
    case handles.Constants.Transformation, comments = 'This is a transformation.';
%       selection_name = Transformations{idx.ProxyIndex}.Name;
%       comments = safeget(Transformations{idx.ProxyIndex}, 'Comments', '');
  end
  set(handles.eFileName, 'String', '-----------------------');
  set(handles.cbVisible, 'Value', 0)
end

set(handles.pbSelectFile, 'Enable', ctrl_browser);
set(handles.pbAddFile, 'Enable', ctrl_addmaster);
set(handles.pbAddProxyFile, 'Enable', ctrl_addslave);
set(handles.eFileName, 'Enable', ctrl_filename);
set(handles.pbLoadDataToImage, 'Enable', ctrl_reload);
set([handles.pbSetName, handles.pbRemoveFile, handles.pbChangeFile], 'Enable', ctrl_set_name);
set(handles.pmImageType, 'Enable', ctrl_imtype);
set(handles.ToolBar(handles.ToolBarGray), 'Enable', enable_viewer)
set(handles.eComment, 'String', comments)
set(handles.eImageTitle, 'String', selection_name);

% --------------------------------------------------------------------
function pbSetComment_Callback(hObject, eventdata, handles) 
pos = get(handles.lbObjects, 'Value');
if isempty(pos), return; end

str = get(handles.eComment, 'String');
idx = handles.ListBoxIndex{pos};
switch idx.ImageIndex
  %   case handles.Constants.Group_2D, set(handles.pmImageType, 'Value', 1);
  %   case handles.Constants.Group_3D, set(handles.pmImageType, 'Value', 2);
  %   case handles.Constants.Group, set(handles.eComment, 'String', '');
  %   case handles.Constants.Coordinate, set(handles.eComment, 'String', '');
  %   case handles.Constants.Sequence, set(handles.eComment, 'String', '');
  case handles.Constants.Transformation
    handles.Transformations{idx.ProxyIndex}.Comments = str;
end
guidata(hObject, handles);

% --------------------------------------------------------------------
function pbSetName_Callback(hObject, eventdata, handles) 
pos = get(handles.lbObjects, 'Value');
if isempty(pos), return; end

str = get(handles.eImageTitle, 'String');
idx = handles.ListBoxIndex{pos};
switch idx.ImageIndex
  case handles.Constants.Folder
%     set(handles.pmImageType, 'Value', 1);
  case handles.Constants.Group_2D 
%     set(handles.pmImageType, 'Value', 1);
  case handles.Constants.Group_3D 
%     set(handles.pmImageType, 'Value', 2);
  case handles.Constants.Transformation
    handles.Transformations{idx.ProxyIndex}.Name = str;
  case handles.Constants.Group
%     set(handles.eComment, 'String', '');
  case handles.Constants.Coordinate
%      set(handles.eComment, 'String', '');
  case handles.Constants.Sequence
%     set(handles.eComment, 'String', '');
  otherwise
    arbuz_SetImage(handles.hGUI, {idx.ImageIndex, idx.ProxyIndex}, 'Name', str);
end

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function pbMainViewer_Callback(hObject, eventdata, handles) 
arbuz_RedrawAll(handles.hGUI)

% --------------------------------------------------------------------
function pbSelectFile_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName, filterindex] = uigetfile(handles.FileTypes,'Open file', old_path, ...
  'MultiSelect', 'on');

if PathName ~= 0
  handles.ini.Directories.SourcePath = PathName;
  if ~iscell(FileName), FileName = {FileName}; end
  
  set(handles.eFileName, 'String', fullfile(PathName, FileName{1}));
  switch filterindex
    case 1 % JPEG file
      new_object.ImageType = '2D';
      set(handles.pmImageType, 'Value', 1)
    case 2 % TIFF file
      new_object.ImageType = '2D';
      set(handles.pmImageType, 'Value', 1)
  end
  guidata(handles.hGUI, handles);
end

% --------------------------------------------------------------------
function [new_image, slave_images] = LoadImage(handles, new_image)

set(handles.hGUI,'Pointer','watch');drawnow
if ~exist(new_image.FileName, 'file')
  disp('File was not found - requesting a new one');
  [~,~,ext] = fileparts(new_image.FileName);
  filter = {['*',ext], ['Extesion (*',ext,')']; '*.*','All Files(*.*)'};
  [file,path] = uigetfile(filter, 'Select file', new_image.FileName);
  if ~isempty(file)
    new_image.FileName = fullfile(path, file);
  else
    return;
  end
end
[new_image.data, new_image.data_info,new_image.ImageType, slave_images] = ...
    arbuz_LoadImage(new_image.FileName, new_image.ImageType);
new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
new_image.isLoaded = 1;
new_image.A = eye();
new_image.Aprime = eye(4);
set(handles.hGUI,'Pointer','arrow'); drawnow

if strcmp(new_image.ImageType, 'GENERIC')
  new_image.ImageType = new_image.data_info.generic_type;
end

if strcmp(new_image.ImageType, 'WORKSPACE') || strcmp(new_image.ImageType, 'MAT-GENERAL')
  if ndims(new_image.data) == 2
    new_image.ImageType = '2D';
  else
    new_image.ImageType = 'FITRESULT';
  end
end

% --------------------------------------------------------------------
function pbAddFile_Callback(hObject, eventdata, handles) 

new_image = GetGUIImageData(handles);

new_image = LoadImage(handles, new_image);

arbuz_ShowMessage(handles.hGUI, sprintf('%s is loaded.', new_image.FileName));
arbuz_AddImage(handles.hGUI, new_image);
output_list = arbuz_FindImage(handles.hGUI, 'master', 'name', new_image.Name, {'Name', 'Anative'});
if ~isempty(output_list)
  arbuz_SetTransformation(handles.hGUI, 'T1', output_list{1}.Name, output_list{1}.Anative);
end

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function pbAddProxyFile_Callback(hObject, eventdata, handles)


% new_image = LoadImage(handles, new_image);
% 
% 
% arbuz_ShowMessage(handles.hGUI, sprintf('%s is loaded.', new_image.FileName));
% arbuz_AddImage(handles.hGUI, new_image);

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function pbDefaultViewer_Callback(handles, button_pressed) 

pos = get(handles.lbObjects, 'Value');
if isempty(pos), return; end

idx = handles.ListBoxIndex{pos};

switch idx.ImageIndex
  %   case handles.Constants.Group_2D, set(handles.pmImageType, 'Value', 1);
  %   case handles.Constants.Group_3D, set(handles.pmImageType, 'Value', 2);
  %   case handles.Constants.Group, set(handles.eComment, 'String', '');
  %   case handles.Constants.Coordinate, set(handles.eComment, 'String', '');
  case handles.Constants.Sequence, disp(handles.Sequences{idx.ProxyIndex});
  case handles.Constants.Transformation
    TransformationShow(handles, idx.ProxyIndex);
  otherwise
    options.image.ImageIdx = idx.ImageIndex;
    options.image.SlaveIdx = idx.ProxyIndex;
    arbuz_DefaultRDR(handles.hGUI, options);
end

% --------------------------------------------------------------------
function pbRemoveFile_Callback(hObject, eventdata, handles)
mImageDelete_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbChangeFile_Callback(hObject, eventdata, handles)

pos = get(handles.lbObjects, 'Value');
if length(pos) > 1, return; end

iidx = handles.ListBoxIndex{pos}.ImageIndex;

if iidx >= 1
  new_image = GetGUIImageData(handles);
  new_image.A = handles.images{iidx}.A;
  if new_image.isStore || new_image.Visible
    [new_image.data, new_image.data_info] = ...
      arbuz_LoadImage(new_image.FileName, new_image.ImageType);
    new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
    new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
    new_image.isLoaded = 1;
  else
    new_image.data = [];
    [new_image.box, new_image.data_info] = LoadImageParameters(new_image.FileName, new_image.ImageType);
    new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
    new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
    new_image.isLoaded = 0;
  end
  new_image.proxy = {};
  handles.images{iidx} = new_image;
  
  % Update object list index
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function cbVisible_Callback(hObject, eventdata, handles) 
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {});

val  = get(handles.cbVisible, 'Value');
arbuz_SetImage(handles.hGUI, find_list, 'Visible', val);

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function pbColor_Callback(hObject, eventdata, handles) 
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'NotSelectedColor', 'SelectedColor'});

if ~isempty(find_list)
  color = SetColorDLG(find_list{1}.NotSelectedColor);
  arbuz_SetImage(handles.hGUI, find_list, 'SelectedColor', color);
  arbuz_SetImage(handles.hGUI, find_list, 'NotSelectedColor', color);
  
  % Update object list index
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function cbStoreInProject_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'FileName'});

val  = get(handles.cbStoreInProject, 'Value');

short_list = {};
for ii=1:length(find_list)
  if ~isempty(find_list{ii}.FileName)
    short_list{end+1} = find_list{ii};
  end
end

if length(find_list) ~= length(short_list)
  arbuz_ShowMessage(handles.hGUI, 'ArbuzGUI: only images that loaded from files can change the storage status.');
end

arbuz_SetImage(handles.hGUI, short_list, 'isStore', val);


% --------------------------------------------------------------------
function handles = AddRecentProjectList(handles, fname)

isfound = 0;
for ii=1:length(handles.RecentProjectList)
  if strcmp(handles.RecentProjectList{ii}.fname, fname), isfound = 1; break; end
end

if ~isfound
  for ii = 1:length(handles.RecentProjectList)
    delete(handles.RecentProjectList{ii}.hmenu);
  end
  handles.RecentProjectList{end+1}.fname = fname;
  handles.RecentProjectList = {handles.RecentProjectList{[max(1, end-15):end]}};
  
  for ii = 1:length(handles.RecentProjectList)
    handles.RecentProjectList{ii}.hmenu = uimenu(handles.mFile, ...
      'Label', epr_ShortFileName(handles.RecentProjectList{ii}.fname, 30), ...
      'Callback', 'ArbuzGUI(''OpenProject_Callback'',gcbo,guidata(gcbo))', ...
      'Tag', 'RecentProject', 'UserData', ii);
    if ii == 1, set(handles.RecentProjectList{ii}.hmenu, 'Separator', 'on'); end
  end
end
guidata(handles.hGUI, handles);

% --------------------------------------------------------------------
function OpenProject_Callback(hObject, handles)
fname = handles.RecentProjectList{get(hObject, 'UserData')}.fname;
OpenProject(fname, handles);

% --------------------------------------------------------------------
function mFileSaveProject_Callback(hObject, eventdata, handles)
if ~isempty(arbuz_get(handles.hGUI, 'FileName'))
  FileName = arbuz_get(handles.hGUI, 'FileName');
  SaveProject(FileName, handles)
  UpdateListbox([], guidata(handles.hGUI));
else
  mFileSaveProjectAs_Callback(hObject, eventdata, handles);
  UpdateListbox([], guidata(handles.hGUI));
end

% --------------------------------------------------------------------
function mFileSaveProjectAs_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'ProjectPath', 'C:/');

[FileName,PathName] = uiputfile({...
  '*.mat', 'Matlab project files (*.mat)'},'Open file', old_path);

if PathName ~= 0
  handles.ini.Directories.ProjectPath = PathName;
  FileName = fullfile(PathName, FileName);
  arbuz_set(handles.hGUI, 'FileName', FileName);
  SaveProject(FileName, handles)
  guidata(handles.hGUI, handles);
end

% --------------------------------------------------------------------
function mHelpAbout_Callback(hObject, eventdata, handles) 

msgbox({'ArbuzGUI';'3D image registration toolbox';...
  'Version 2.2 (December 2015)'; '';...
  'Author: Boris Epel';
  'Transformation routines: Charles Pelizzari'; '';...
  'University of Chicago';...
  'epri.uchicago.edu';...
  'Center for EPR Imaging in vivo Physiology';...
  'Department of Radiation and Cellular Oncology'; 'Copyright 2009 - 2015.'}, ...
  'About ArbuzGUI', 'custom', handles.ico.arbuz);

FileName = arbuz_get(handles.hGUI, 'FileName');
fprintf('---------------\nLoading script:\n');
fprintf('hGUI=figure(1000); arbuz_OpenProject(hGUI, ''%s'');\n', FileName);
fprintf('output_list = arbuz_FindImage(hGUI, ''master'', ''NAME'', ''MRI'', {''slavelist''});\n');
fprintf('assert(~isempty(MRI_list), ''No MRI entry found'');\n'); 
fprintf('MRI = MRI_list{1}\n');
fprintf('---------------\n');

% --------------------------------------------------------------------
function mFileExit_Callback(hObject, ~, handles) 
closereq;

% --------------------------------------------------------------------
function mFileNewProject_Callback(hObject, eventdata, handles) 
arbuz_InitializeProject(handles.hGUI);

% Update object list index
UpdateListbox(0,handles);


% --------------------------------------------------------------------
function mFileOpenProject_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'ProjectPath', 'C:/');

[FileName,PathName] = uigetfile({...
  '*.mat', 'Matlab project files (*.mat)'},'Project file', old_path);

if PathName ~= 0
  handles.ini.Directories.ProjectPath = PathName;
  
  % OpenProject saves handles by itself
  OpenProject(fullfile(PathName, FileName), handles);
  handles = guidata(handles.MainFigure);
  AddRecentProjectList(handles, fullfile(PathName, FileName));
end

% --------------------------------------------------------------------
function mFileAddToProject_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'ProjectPath', 'C:/');

[FileName,PathName] = uigetfile({...
  '*.mat', 'Matlab project files (*.mat)'},'Project file', old_path);

if PathName ~= 0
  % Fix current transformation
  arbuz_ApplyTransformation(handles.hGUI, '', 'fix');
  
  h = figure(1010);
  arbuz_OpenProject(h, fullfile(PathName, FileName));
  % Fix current transformation
  arbuz_ApplyTransformation(h, '', 'fix');

  prj = getappdata(h, 'project');
  isOK = false;
  while ~isOK
    % request suffix
    a = inputdlg('Suffix:', 'Suffix for the images in the new project', 1, {'2'});
    if isempty(a), delete(h); return; end
    
    % verify the suffix
    isOK = true;
    suffix = ['_',a{1}];
    for ii=1:length(prj.images)
      res = arbuz_FindImage(handles.hGUI, 'all', 'Name', [prj.images{ii}.Name,suffix], {});
      if ~isempty(res)
        isOK = false; break;
      end
    end
    if ~isOK, continue; end
    
    % add images
    for ii=1:length(prj.images)
      prj.images{ii}.Name = [prj.images{ii}.Name, suffix];
      arbuz_AddImage(handles.hGUI, prj.images{ii});
    end
    
    % add transformations
    for ii=1:length(prj.Transformations)
      arbuz_AddTransformation(handles.hGUI, prj.Transformations{ii}.Name);
      
      for jj=1:length(prj.Transformations{ii}.Matrices)
        arbuz_SetTransformation(handles.hGUI, prj.Transformations{ii}.Name, ...
          [prj.Transformations{ii}.Matrices{jj}.Image,suffix], ...
          prj.Transformations{ii}.Matrices{jj}.A);
      end
    end
    
    arbuz_UpdateInterface(handles.hGUI);
  end
  
  delete(h);
end

% --------------------------------------------------------------------
function LaunchPlugin(handles, plugin_name)
% plugin  = fullfile(handles.ini.Directories.PluginsPath,[plugins{pos}, 'PLG']);
plugin  = [plugin_name, 'PLG(handles.hGUI)'];
hPlugIns = eval(plugin);

handles.ActivePlugins(end+1) = hPlugIns;
guidata(handles.hGUI, handles);

try eval([plugin,'(''UpdateSelection'',hPlugIns, guidata(hPlugIns));']); catch end

% --------------------------------------------------------------------
function LaunchReport(handles, plugin_name)
% plugin  = fullfile(handles.ini.Directories.PluginsPath,[plugins{pos}, 'PLG']);
plugin  = [plugin_name, 'RPT(handles.hGUI)'];
hPlugIns = eval(plugin);

handles.ActiveReports(end+1) = hPlugIns;
guidata(handles.hGUI, handles);

try eval([plugin,'(''UpdateSelection'',hPlugIns, guidata(hPlugIns));']); catch end

% --------------------------------------------------------------------
function mPluginReloadAction_Callback(hObject, eventdata, handles)

handles.ini.Directories.PluginsPath = safeget(handles.ini.Directories, 'PluginsPath', fileparts(which('ArbuzGUI')));
plugins = dir(fullfile(handles.ini.Directories.PluginsPath, '*PLG.fig'));

str = {};
for ii=1:length(plugins)
  [~, fname] = fileparts(plugins(ii).name);
  fname = regexp(fname, '(?<name>.+)PLG', 'names');
  str{end+1} = fname.name;
end

obj = findobj(handles.mPlugin);
obj = obj(obj~=handles.mPluginReloadAction & obj~=handles.mPlugin);
delete(obj);

for ii=1:length(str)
  uimenu(handles.mPlugin, 'Label', str{ii}, 'Separator', iff(ii==1, 'on', 'off'), ...
    'Callback',sprintf('ArbuzGUI(''LaunchPlugin'', guidata(gcbo), ''%s'')', str{ii}));
end

handles.ActivePlugins = [];

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function mReportsReloadReports_Callback(hObject, eventdata, handles)
handles.ini.Directories.PluginsPath = safeget(handles.ini.Directories, 'PluginsPath', fileparts(which('ArbuzGUI')));
plugins = dir(fullfile(handles.ini.Directories.PluginsPath, '*RPT.fig'));

str = {};
for ii=1:length(plugins)
  [~, fname] = fileparts(plugins(ii).name);
  fname = regexp(fname, '(?<name>.+)RPT', 'names');
  str{end+1} = fname.name;
end

obj = findobj(handles.mReports);
obj = obj(obj~=handles.mReports & obj~=handles.mReportsReloadReports);
delete(obj);

for ii=1:length(str)
  uimenu(handles.mReports, 'Label', str{ii}, 'Separator', iff(ii==1, 'on', 'off'), ...
    'Callback',sprintf('ArbuzGUI(''LaunchReport'', guidata(gcbo), ''%s'')', str{ii}));
end
handles.ActiveReports = [];

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ------- S E Q U E N C E --------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
function mSequenceActive_Callback(hObject, eventdata, handles)
Sequences = arbuz_get(handles.hGUI, 'Sequences');
str = cell(1,length(Sequences));
for ii=1:length(Sequences)
  str{ii} = Sequences{ii}.Name;
end

[idx] = listdlg('Name', 'Select', 'PromptString', 'Select sequence', ...
  'ListSize', [160 120], ...
  'SelectionMode', 'single', 'ListString', str);

if ~isempty(idx)
  arbuz_set(handles.hGUI, 'ActiveSequence', idx);
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mSequenceEdit_Callback(hObject, eventdata, handles) 

ActiveSequence = arbuz_get(handles.hGUI, 'ActiveSequence');

if ActiveSequence > 0
  [modified_sequence, isOk] = EditSequenceDLG(handles.hGUI, ActiveSequence);
  if isOk
    Sequences = arbuz_get(handles.hGUI, 'Sequences');
    Sequences{ActiveSequence} = modified_sequence;
    arbuz_set(handles.hGUI, 'Sequences', Sequences);

    str = {};
    for jj = 1:length(modified_sequence.Sequence)
      str{end+1} = ['Watch after stage: ', modified_sequence.Sequence{jj}.Name];
    end
    enable_control = 'on'; pos = modified_sequence.WatchTransformation;
    if isempty(str), str = {'Watch: nothing'}; enable_control = 'off'; pos = 1; end
    set(handles.pmSeeStage, 'String', str, 'Value', pos, 'Enable', enable_control);
  end
end
UpdateListbox(0,guidata(handles.hGUI));

% --------------------------------------------------------------------
function mSequenceNew_Callback(hObject, eventdata, handles) 
answer = inputdlg('Type the name of new sequence', 'Input', 1, {'sequence'});

if ~isempty(answer)
  arbuz_AddSequence(handles.hGUI, answer{1});
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mSequenceDelete_Callback(hObject, eventdata, handles) 
Sequences = arbuz_get(handles.hGUI, 'Sequences');
str = cell(1,length(Sequences));
for ii=1:length(Sequences)
  str{ii} = Sequences{ii}.Name;
end

[idx] = listdlg('Name', 'Select', 'PromptString', 'Select sequence', ...
  'ListSize', [160 120], ...
  'SelectionMode', 'single', 'ListString', str);

if ~isempty(idx)
  arbuz_DeleteSequence(handles.hGUI, str{idx});
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mSequenceDeleteAll_Callback(hObject, eventdata, handles) 
arbuz_DeleteSequence(handles.hGUI);
UpdateListbox(0,handles);

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ------- T R A N S F O R M A T I O N --------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
function mTransformationNew_Callback(hObject, eventdata, handles) 
answer = inputdlg('Type the name of new transformation', 'Input', 1, {'transformation'});

if ~isempty(answer)
  arbuz_AddTransformation(handles.hGUI, answer{1});
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mTransformationSave_Callback(hObject, eventdata, handles)
current_transformation = arbuz_get(handles.hGUI, 'ActiveTransformationName');
arbuz_StageToTransformation(handles.hGUI, current_transformation);

% --------------------------------------------------------------------
function mTransformationSaveAs_Callback(hObject, eventdata, handles)
answer = inputdlg('Type the name of new transformation', 'Input', 1, {'transformatio'});

if ~isempty(answer)
  handles = arbuz_AddTransformation(handles, answer{1});
end

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function mTransformationDelete_Callback(hObject, ~, handles)
disp('ArbuzGUI: Not implemented.');

% --------------------------------------------------------------------
function mTransformationReset_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {});

transformation = arbuz_GetActiveTransformation(handles.hGUI);
arbuz_SetImage(handles.hGUI, find_list, 'A', eye(4));
arbuz_SetImage(handles.hGUI, find_list, 'Aprime', eye(4));
arbuz_SetTransformation(handles.hGUI, transformation, find_list{1}.Image, eye(4));

arbuz_SetActiveTransformation(handles.hGUI);
arbuz_RedrawAll(handles.hGUI);

% --------------------------------------------------------------------
function mTransformationResetRotation_Callback(hObject, ~, handles)
disp('ArbuzGUI: Not implemented.');

% --------------------------------------------------------------------
function mTransformationResetTranslation_Callback(hObject, eventdata, handles)

for ii=1:length(handles.images)
  handles.images{ii}.A(4,1:3) = 0;
  handles.images{ii}.Aprime = eye(4);
end
guidata(handles.hGUI, handles);

% --------------------------------------------------------------------
function mTransformationResetScaling_Callback(hObject, ~, handles)
disp('ArbuzGUI: Not implemented.');

% --------------------------------------------------------------------
function mImageAdd_Callback(hObject, eventdata, handles) 
arbuz_ShowMessage(handles.hGUI, 'ArbuzGUI: To add image select ''2D images''/''3D images'' group and follow instructions.');

% --------------------------------------------------------------------
function mImageDelete_Callback(hObject, eventdata, handles)
pos = get(handles.lbObjects, 'Value');

delete_images = {};
for ii = 1:length(pos)
  image_idx = handles.ListBoxIndex{pos(ii)}.ImageIndex;
  proxy_idx = handles.ListBoxIndex{pos(ii)}.ProxyIndex;
  im = arbuz_FindImage(handles.hGUI, {image_idx, proxy_idx}, '', '', {'Name'});
  delete_images{end+1} = im{1};
end

for ii = 1:length(delete_images)
  if ~isempty(delete_images{ii})
    str = sprintf('You are about to delete ''%s''.  Are you sure ?', delete_images{ii}.Name);
    button = questdlg(str, 'Warning', 'Yes' ,'No', 'No');
    if strcmp(button, 'Yes')
      im = arbuz_FindImage(handles.hGUI, {delete_images{ii}.Image, delete_images{ii}.Slave}, '', '', {'Name'});
      arbuz_DeleteImage(handles.hGUI, im);
    end
  end
end

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function TransformationShow(handles, ShowTrans)

Transformations = arbuz_get(handles.hGUI, 'Transformations');
for ii=1:length(ShowTrans)
  if ShowTrans(ii) > 0
    T = Transformations{ShowTrans(ii)};
    disp(['Transformation: ', T.Name])
    for jj=1:length(T.Matrices)
      disp(T.Matrices{jj}.Image)
      disp(T.Matrices{jj}.A)
    end
  end
end

% --------------------------------------------------------------------
function mTransformationShow_Callback(hObject, eventdata, handles)

Transformations = arbuz_get(handles.hGUI, 'Transformations');
str = {};
for ii=1:length(Transformations), str{end+1}=Transformations{ii}.Name; end
ShowTrans = listdlg('ListString',str, 'PromptString', 'SelectTransformation', 'ListSize', [160, 140]);
TransformationShow(handles, ShowTrans);

% --------------------------------------------------------------------
function mImageExportWorkspace_Callback(hObject, eventdata, handles)
pos = get(handles.lbObjects, 'Value');
if length(pos) > 1, pos = pos(1); end

image_idx = handles.ListBoxIndex{pos}.ImageIndex;
proxy_idx = handles.ListBoxIndex{pos}.ProxyIndex;

res = arbuz_FindImage(handles.hGUI, {image_idx, proxy_idx}, '', '', {'data', 'FullName', 'SlaveList'});
if isempty(res), return; end
assignin('base', 'arbuz_image', res{1});
disp(sprintf('Image ''%s'' is exported. Variable ''arbuz_image'' is created.', res{1}.FullName));
evalin('base', 'disp(arbuz_image)');

% --------------------------------------------------------------------
function mImageCopyAllTrans_Callback(hObject, eventdata, handles)
pos = get(handles.lbObjects, 'Value');
image_dest.ImageIdx = handles.ListBoxIndex{pos}.ImageIndex;
image_dest.SlaveIdx = handles.ListBoxIndex{pos}.ProxyIndex;
image_dest = arbuz_FindImage(handles.hGUI, image_dest, '', '', {});

if image_dest{1}.ImageIdx < 1; return; end
if image_dest{1}.SlaveIdx >= 1
  disp('This options is working on master images only');
  return;
end

str = arbuz_get(handles.hGUI, 'IMAGENAMES');
idx = listdlg('ListString',str, 'PromptString', 'Select image to copy FROM', 'ListSize', [160, 140]);
image_src.ImageIdx  = idx;
image_src.SlaveIdx = -1;
image_src = arbuz_FindImage(handles.hGUI, image_src, '', '', {});

arbuz_ApplyTransformation(handles.hGUI, '', 'fix');
if hObject == handles.mImageCopyAllTrans
  arbuz_CopyAllTransformations(handles.hGUI, image_src{1}, image_dest{1}, 2);
else
  arbuz_CopyAllTransformations(handles.hGUI, image_src{1}, image_dest{1}, 1);
end

% --------------------------------------------------------------------
function mImageShow_Callback(hObject, eventdata, handles) 
pos = get(handles.lbObjects, 'Value');
image_idx = handles.ListBoxIndex{pos}.ImageIndex;
proxy_idx = handles.ListBoxIndex{pos}.ProxyIndex;

DefaultViewer(handles, image_idx, proxy_idx);

% --------------------------------------------------------------------
function mImageResetMatrix_Callback(hObject, eventdata, handles) 
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {});
find_list = find_list(find_list{:}.SlaveIdx < 0);

arbuz_SetImage(handles.hGUI, find_list, 'A', eye(4));
arbuz_SetImage(handles.hGUI, find_list, 'Aprime', eye(4));

% Update object list index
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function mImageShowTransformation_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'Aslave', 'Apre', 'A', 'Aprime', 'Acurrent', 'Anext','FullName','Ashow','bbox'});
if isempty(find_list), return; end

fprintf('\n-----------------\nImage: ''%s''.\n', find_list{1}.FullName);
fprintf('Size: %i %i %i %i.\n', find_list{1}.Box);
if(strcmp(find_list{1}.ImageType,'3DMASK'))
  find_list2 = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'Data', 'ANative','box'});
  Vox = diag(find_list2{1}.Anative);
  nVox = numel(find(find_list2{1}.data));
  fprintf('Volume: %5.3f cm^3 (%i voxels)\n', nVox*prod(Vox(1:3))*1E-3, nVox); 
end

fprintf('Non-unit transformations:\n');
Transformations = arbuz_get(handles.hGUI, 'Transformations');
for ii=1:length(Transformations)
  for jj=1:length(Transformations{ii}.Matrices)
    if strcmp(Transformations{ii}.Matrices{jj}.Image, find_list{1}.Image)
      disp(['Transformation: ',Transformations{ii}.Name])
      disp(Transformations{ii}.Matrices{jj}.A)
    end
  end
end

if find_list{1}.SlaveIdx >= 1
  disp('Aslave'); disp(find_list{1}.Aslave);
end

if sum(sum(abs(find_list{1}.Apre-eye(4)))) > 1E-10
  disp('Aprevious'); disp(find_list{1}.Apre);
end
if sum(sum(abs(find_list{1}.A-eye(4)))) > 1E-10
  disp('A'); disp(find_list{1}.A);
end
if sum(sum(abs(find_list{1}.Aprime-eye(4)))) > 1E-10
  disp('Aprime'); disp(find_list{1}.Aprime);
end
disp('Acurrent'); disp(find_list{1}.Acurrent);
if sum(sum(abs(find_list{1}.Anext-eye(4)))) > 1E-10
  disp('Anext'); disp(find_list{1}.Anext);
end
disp(['Scale: ', num2str(1./hmatrix_scale_get(find_list{1}.Ashow))])




% --------------------------------------------------------------------
function pbImageInfo_Callback(hObject, eventdata, handles)
pos = get(handles.lbObjects, 'Value');
if isempty(pos), return; end

idx = handles.ListBoxIndex{pos};
switch idx.ImageIndex
  %   case handles.Constants.Group_2D, set(handles.pmImageType, 'Value', 1);
  %   case handles.Constants.Group_3D, set(handles.pmImageType, 'Value', 2);
  %   case handles.Constants.Group, set(handles.eComment, 'String', '');
  case handles.Constants.Coordinate 
    Coordinates = arbuz_get(handles.hGUI, 'Coordinates');
    if ~isempty(Coordinates)
      disp(Coordinates{1}.Name)
      disp(Coordinates{1}.A)
    end
  %   case handles.Constants.Sequence, set(handles.eComment, 'String', '');
  case handles.Constants.Transformation
    TransformationShow(handles, idx.ProxyIndex);
  otherwise
    mImageShowTransformation_Callback(hObject, eventdata, handles)
end

% --------------------------------------------------------------------
function mViewMultyImageAction_Callback(hObject, eventdata, handles)
SelectImageDLG(handles.hGUI)

% --------------------------------------------------------------------
function mImageMakeMaster_Callback(hObject, ~, handles)
disp('ArbuzGUI: Not implemented.');

% --------------------------------------------------------------------
function mImageMakeProxy_Callback(hObject, ~, handles)
disp('ArbuzGUI: Not implemented.');

% --------------------------------------------------------------------
function pbLoadDataToImage_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'Name', 'FileName'});

for ii=1:length(find_list)
  [new_image, slave_images] = LoadImage(handles, find_list{ii});
  arbuz_SetImage(handles.hGUI, find_list{ii}, 'data', new_image.data);
  arbuz_SetImage(handles.hGUI, find_list{ii}, 'filename', new_image.FileName);
  arbuz_SetImage(handles.hGUI, find_list{ii}, 'Anative', new_image.Anative);
  if isfield(new_image.data_info, 'Mask') && ~isempty(new_image.data_info.Mask)
    arbuz_SetImage(handles.hGUI, find_list{ii}, 'data_info_mask', new_image.data_info.Mask);
  end

  for jj=1:length(slave_images)
    arbuz_AddImage(handles.hGUI, slave_images{jj}, find_list{ii}.Image);
  end
end

UpdateListbox(0,handles);

% --------------------------------------------------------------------
function mViewApplyGroupDefault_Callback(hObject, eventdata, handles) 
Groups = arbuz_get(handles.hGUI, 'Groups');

if ~isempty(Groups)
  str = cell(length(Groups), 1);
  for ii=1:length(Groups), str{ii} = Groups{ii}.Name; end
  grp_idx = listdlg('ListString',str, 'PromptString', 'Select group', 'ListSize', [160, 140]);
  
  if ~isempty(grp_idx)
    % clear all settings
    find_list = arbuz_FindImage(handles.hGUI, 'all', '', '', {});
    arbuz_SetImage(handles.hGUI, find_list, 'Visible', 0);
    
    arbuz_SetImage(handles.hGUI, Groups{grp_idx}.list, 'Visible', 1);
    %   for ii=1:length(handles.Groups{grp_idx}.list)
    %     handles = arbuz_SetImage(handles, handles.Groups{grp_idx}.list, ...
    %       'Color', handles.Groups{grp_idx}.list{ii}.Color);
    %   end
    
    % Update object list index
    UpdateListbox(0,handles);
  end
end

% --------------------------------------------------------------------
function mViewSaveGroupAs_Callback(hObject, eventdata, handles)

answer = inputdlg('The name of new group', 'Input', 1, {'Group1'});

if ~isempty(answer)
  find_list = arbuz_FindImage(handles.hGUI, {}, 'highlighted', 0, {});
  arbuz_AddGroup(handles.hGUI, answer{1}, find_list);
  
  % Update object list index
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mViewSaveGroup_Callback(hObject, eventdata, handles) 
Groups = arbuz_get(handles.hGUI, 'Groups');

if isempty(Groups)
  mViewSaveGroupAs_Callback(hObject, eventdata, handles);
else
  find_list = arbuz_FindImage(handles.hGUI, {}, 'highlighted', 0, {});
  
  str = {};
  for ii=1:length(Groups), str{ii} = Groups{ii}.Name; end
  grp_idx = listdlg('ListString',str, 'PromptString', 'Select group', 'ListSize', [160, 140]);
  
  if ~isempty(grp_idx)
    arbuz_AddGroup(handles.hGUI, Groups{grp_idx}.Name, find_list);
    % Update object list index
    UpdateListbox(0,handles);
  end
end

% --------------------------------------------------------------------
function mViewDeleteGroup_Callback(hObject, eventdata, handles)
Groups = arbuz_get(handles.hGUI, 'Groups');

if ~isempty(Groups)
  str = {};
  for ii=1:length(Groups), str{ii} = Groups{ii}.Name; end
  grp_idx = listdlg('ListString',str, 'PromptString', 'Select group', 'ListSize', [160, 140]);
  
  if ~isempty(grp_idx)
    Groups = Groups([1:grp_idx-1, grp_idx+1:end]);
    arbuz_ShowMessage(handles.hGUI, 'ArbuzGUI: group is deleted.');
    arbuz_set(handles.hGUI, 'Groups', Groups);

    % Update object list index
    UpdateListbox(0,handles);
  end
end

% --------------------------------------------------------------------
function mViewSaveGroupScene_Callback(hObject, eventdata, handles)

new_save.renderer = 'arbuz_3DRDR';
new_save.plugin_name = 'arbuz_3DRDR';
new_save.FigN = fix(str2double(get(handles.eViewerFigure, 'string')));

new_save.images = arbuz_FindImage(handles.MainFigure, 'v', '', '', {});

% find transformation that leads to the frame of interest
val = get(handles.pmReferenceFrame, 'Value');
str = get(handles.pmReferenceFrame, 'String');
if val == 1
  new_save.A2frame = eye(4);
else
  find_origin = arbuz_FindImage(handles.MainFigure, 'all', 'FullName', str{val}, {'Ashow'});
  if isempty(find_origin), return; end
  new_save.A2frame = inv(find_origin{1}.Ashow);
end

save_pars = arbuz_GetSaveParList(handles.MainFigure, new_save.renderer);
if isempty(save_pars)
  answer=inputdlg({'Parameter''s name'}, 'Input the name', 1, {new_save.plugin_name});
  if ~isempty(answer)
    arbuz_SetSavePar(handles.MainFigure, answer, new_save.renderer, new_save);
  end
else
  save_pars = [{'New name'}, save_pars{:}];
  [sel,isOk] = listdlg('PromptString','Select an image:',...
    'SelectionMode','single',...
    'ListString',save_pars);
  if isOk
    if sel == 1
      new_name=inputdlg({'Parameter''s name'}, 'Input the name', 1, {'AlphaSlice'});
      new_name = new_name{1};
    else
      new_name = save_pars{sel};
    end
    arbuz_SetSavePar(handles.MainFigure, new_name, new_save.renderer, new_save);
  end
end
% --------------------------------------------------------------------
function mImageSetNativeTrans_Callback(hObject, eventdata, handles) 
find_list = arbuz_FindImage(handles, 'master', 'Highlighted', 1, {'Name', 'Anative'});

for ii=1:length(find_list)
  handles = arbuz_SetImage(handles, {find_list{ii}}, 'Aprime', find_list{ii}.Anative);
end

% --------------------------------------------------------------------
function pbBringUpFigure_Callback(hObject, eventdata, handles) 
figure(str2double(get(handles.eViewerFigure, 'String')));

% --------------------------------------------------------------------
function mImageSave_Callback(hObject, eventdata, handles)
pos = get(handles.lbObjects, 'Value');
if length(pos) > 1, pos = pos(1); end
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

img = arbuz_FindImage(handles.hGUI, {handles.ListBoxIndex{pos}.ImageIndex, handles.ListBoxIndex{pos}.ProxyIndex}, '', '', ...
  {'data';'bbox';'Anative'});

if ~isempty(img)
  img = img{1};
  switch img.ImageType
    case '3DMASK'
      [FileName,PathName, filterindex] = uiputfile({...
        '*.mat', 'Arbuz generic file type (*.mat)';...
        '*.mat', 'Image mask file (*.mat)';...
        },'Open file', old_path);
      if ~isequal(FileName,0) && ~isequal(PathName,0)
        fname = fullfile(PathName, FileName);
        switch filterindex
          case 1 % Generic file
            img.file_type = 'ArbuzGeneric_v1.0';
            save(fname, '-struct', 'img');
          case 2 % Image mask
            s.Mask = img.data;
            s.file_type = 'ImageMask_v1.0'; 
            save(fname, '-struct', 's');
        end
        fprintf('Image %s is saved.\n',fname);
      end
    otherwise
      [FileName,PathName,selection] = uiputfile(...
        {'*.mat', 'Arbuz generic file type (*.mat)';...
        '*.dcm', 'DICOM (*.dcm)';...
        '*.img', 'AmiraMesh (*.img)';...
        '*.img', 'Binary (*.img)'},...
        'Open file', old_path);
      if ~isequal(FileName,0) && ~isequal(PathName,0)
        fname = fullfile(PathName, FileName);
        switch selection
          case 1 % Generic file
            img.file_type = 'ArbuzGeneric_v1.0';
            save(fname, '-struct', 'img');
          case 2 % DICOM file
            save_dicom_image_file(fname, int16(img.data), [])
          case 3 % Amira file            
            dd = diag(img.Anative);
            sz = img.Box(:).*dd([2,1,3]);
            options.BoundingBox = [-sz(1:3)/2, sz(1:3)/2]';
            options.BoundingBox = options.BoundingBox(:)';
            write_amira_image_file(fname, permute(img.data,[2,1,3]), options);
          case 4 % Amira file            
            dd = diag(img.Anative);
            sz = img.Box(:).*dd([2,1,3]);
            options.BoundingBox = [-sz(1:3)/2, sz(1:3)/2]';
            options.BoundingBox = options.BoundingBox(:)';
            write_binary_image_file(fname, permute(img.data,[2,1,3]), options);
        end
        fprintf('Image %s is saved.\n',fname);
      end
  end
end

% --------------------------------------------------------------------
function mImageLoadGeneric_Callback(hObject, eventdata, handles) 
pos = get(handles.lbObjects, 'Value');
if length(pos) > 1, pos = pos(1); end
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

image_idx = handles.ListBoxIndex{pos}.ImageIndex;
proxy_idx = handles.ListBoxIndex{pos}.ProxyIndex;

proxy = arbuz_FindImage(handles.hGUI, {image_idx,proxy_idx}, '', '', {});
master = arbuz_FindImage(handles.hGUI, {image_idx,-1}, '', '', {});

if proxy_idx == -1
  disp('Only proxy image can be loaded'); return
end

[FileName,PathName] = uigetfile({'*.mat', 'Arbuz generic file type (*.mat)';},...
  'Open file', old_path);
if ~isequal(FileName,0) && ~isequal(PathName,0)
  fname = fullfile(PathName, FileName);
  im = load(fname);
  im.Name = im.Slave;
  if strcmp(safeget(im, 'file_type', ''), 'ArbuzGeneric_v1.0')
    arbuz_AddImage(handles.hGUI, im, master{1}.Image);
    arbuz_UpdateInterface(handles.hGUI);
  end
end

% --------------------------------------------------------------------
% --------------------------------------------------------------------
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
% --------------------------------------------------------------------
% --------------------------------------------------------------------

% --------------------------------------------------------------------

function [ListBoxIndex]=UpdateListbox(dummy, handles)

str = {};          % string to show
ListBoxIndex = {}; % global index of listboxes
list_2D      = []; % list of 2D objects
list_3D      = []; % list of 3D objects

output_list = arbuz_FindImage(handles.hGUI, 'master', '', '', {'Name','isLoaded','slavelist','isStore'});

showMasks   = get(handles.pbImSelMa, 'value');
showAnchors = get(handles.pbImSelAn, 'value');
showSurfaces = get(handles.pbImSelSrf, 'value');

% sort the list
for ii=1:length(output_list)
  im_type = safeget(output_list{ii}, 'ImageType', 'none');
  switch im_type
    case {'2D', 'XYZ', 'CONTOUR'}, list_2D(end+1) = ii;
    case {'3D', 'FITRESULT', 'pEPROI', '3DEPRI', 'AMP_pEPRI', 'PO2_pEPRI', '3DSURFACE', ...
        'MRI','3DMASK','RAW','AMIRA3D','DICOM3D','SHAPE3D','IDL','BIN', 'CEST_ML', ''}, list_3D(end+1) = ii;
  end
end

% compose the 2D list
ListBoxIndex{end+1}.ImageIndex = handles.Constants.Group_2D;
str{end+1} = '<html><font color=blue><b>2D images</b></font></html>';
for ii=1:length(list_2D)
  ListBoxIndex{end+1}.ImageIndex = list_2D(ii);
  ListBoxIndex{end}.ProxyIndex = -1;
  item_str = safeget(output_list{list_2D(ii)}, 'Image', 'Image');
  
  item_length = length(item_str);
  if safeget(output_list{list_2D(ii)}, 'Visible', 0)
    item_str = ['<b><font color=red>',item_str,'</font></b>'];
  else
    item_str = ['<b><font color=teal>',item_str,'</font></b>'];
  end
  
  if safeget(output_list{list_2D(ii)}, 'isLoaded', 0)
    item_str = [item_str, ' L'];
    item_length = item_length + 2;
 end
  
  for kk=(length(item_str):18+length(item_str)-item_length), item_str(kk) = '-'; end
  
  if safeget(output_list{list_2D(ii)}, 'Selected', 0)
    str{end+1} = ['<html>+',item_str,'</html>'];
  else
    str{end+1} = ['<html>&nbsp;',item_str,'</html>'];
  end
  
  % slave images
  output_slave_list = arbuz_FindImage(handles.hGUI, output_list{list_2D(ii)}.SlaveList, '', '', {'Name','isLoaded','isStore'});
  for jj=1:length(output_slave_list)
    sub_item = output_slave_list{jj}.Name;
    if output_slave_list{jj}.Visible
      sub_item = ['<b><font color=red>',sub_item,'</font></b>'];
    end
    if strcmp(output_slave_list{jj}.ImageType, '2D') && ...
        (output_slave_list{jj}.isLoaded || ...
        output_slave_list{jj}.isStore)
      sub_item = [sub_item, ' *'];
    end
    if output_slave_list{jj}.Selected
      str{end+1} = ['<html>&nbsp;&nbsp;+',sub_item,'</html>'];
    else
      str{end+1} = ['<html>&nbsp;&nbsp;&nbsp;',sub_item,'</html>'];
    end
    ListBoxIndex{end+1}.ImageIndex = list_2D(ii);
    ListBoxIndex{end}.ProxyIndex = jj;
  end
end

% compose the 3D list
ListBoxIndex{end+1}.ImageIndex = handles.Constants.Group_3D;
str{end+1} = '<html><font color=blue><b>3D images</b></font></html>';
for ii=1:length(list_3D)
  ListBoxIndex{end+1}.ImageIndex = list_3D(ii);
  ListBoxIndex{end}.ProxyIndex = -1;
  item_str = safeget(output_list{list_3D(ii)}, 'Name', 'Image');
  
  item_length = length(item_str);
  if safeget(output_list{list_3D(ii)}, 'Visible', 0)
    item_str = ['<b><font color=red>',item_str,'</font></b>'];
  else
    item_str = ['<b><font color=teal>',item_str,'</font></b>'];
  end
  
  if ~isempty(safeget(output_list{list_3D(ii)}, 'Link', ''))
    item_str = [item_str, ' ->',handles.images{list_3D(ii)}.Link];
  end
  
  if safeget(output_list{list_3D(ii)}, 'isLoaded', 0)
    item_str = [item_str, ' L'];
    item_length = item_length + 2;
    if safeget(output_list{list_3D(ii)}, 'isStore', 0)
      item_str = [item_str, 'S'];
      item_length = item_length + 1;
    end
  end
  
  for kk=(length(item_str):18+length(item_str)-item_length), item_str(end+1) = '-'; end
  
  if safeget(output_list{list_3D(ii)}, 'Selected', 0)
    str{end+1} = ['<html>+',item_str,'</html>'];
  else
    str{end+1} = ['<html>&nbsp;',item_str,'</html>'];
  end
  
  % slave images
  output_slave_list = arbuz_FindImage(handles.hGUI, output_list{list_3D(ii)}.SlaveList, '', '', {'Name','isLoaded'});
  for jj=1:length(output_slave_list)
    ImageType=output_slave_list{jj}.ImageType;
    if(strcmp(ImageType,'3DMASK') && ~showMasks), continue; end
    if(strcmp(ImageType,'XYZ') && ~showAnchors), continue; end
    if(strcmp(ImageType,'3DSURFACE') && ~showSurfaces), continue; end
    sub_item = output_slave_list{jj}.Name;
    if safeget(output_slave_list{jj}, 'Visible', 0)
      sub_item = ['<b><font color=red>',sub_item,'</font></b>'];
    end
    if safeget(output_slave_list{jj}, 'isLoaded', 0) || ...
        safeget(output_slave_list{jj}, 'isStore', 0)
      sub_item = [sub_item, ' *'];
    end
    if safeget(output_slave_list{jj}, 'Selected', 0)
      str{end+1} = ['<html>&nbsp;&nbsp;+',sub_item,'</html>'];
    else
      str{end+1} = ['<html>&nbsp;&nbsp;&nbsp;',sub_item,'</html>'];
    end
    ListBoxIndex{end+1}.ImageIndex = list_3D(ii);
    ListBoxIndex{end}.ProxyIndex = jj;
  end
end

% compose the Groups list
ListBoxIndex{end+1}.ImageIndex = handles.Constants.Folder;
str{end+1} = '<html><font color=blue><b>Groups</b></font></html>';
Groups = arbuz_get(handles.hGUI, 'Groups');
for ii=1:length(Groups)
  ListBoxIndex{end+1}.ImageIndex = handles.Constants.Group;
  ListBoxIndex{end}.ProxyIndex = ii;
  item_str = Groups{ii}.Name;
  str{end+1} = ['<html>&nbsp;&nbsp;',item_str,'</html>'];
end

% compose the Coordinates list
ListBoxIndex{end+1}.ImageIndex = -1;
Coordinates = arbuz_get(handles.hGUI, 'Coordinates');
str{end+1} = '<html><font color=blue><b>Coordinates</b></font></html>';
for ii=1:length(Coordinates)
  ListBoxIndex{end+1}.ImageIndex = handles.Constants.Coordinate;
  ListBoxIndex{end}.ProxyIndex = ii;
  item_str = Coordinates{ii}.Name;
  str{end+1} = ['<html>&nbsp;&nbsp;<font color=red><b>',item_str,'</b></font></html>'];
end

% compose the Sequence list
active_sequence_title = '';
active_transformation_title = '?';
ListBoxIndex{end+1}.ImageIndex = -1;
ListBoxIndex{end}.ProxyIndex = -1;
str{end+1} = '<html><font color=blue><b>Sequences</b></font></html>';
Sequences = arbuz_get(handles.hGUI, 'Sequences');
ActiveSequence = arbuz_get(handles.hGUI, 'ActiveSequence');
for ii=1:length(Sequences)
  ListBoxIndex{end+1}.ImageIndex = handles.Constants.Sequence;
  ListBoxIndex{end}.ProxyIndex = ii;
  item_str = Sequences{ii}.Name;
  if ii == ActiveSequence
    str{end+1} = ['<html>&nbsp;&nbsp;<font color=red><b>',item_str,'</b></font></html>'];
    active_sequence_title = item_str;
    for jj = 1:length(Sequences{ii}.Sequence)
      sub_item = Sequences{ii}.Sequence{jj}.Name;
      if jj == safeget(Sequences{ii}, 'WatchTransformation', -1)
        sub_item = ['<font color=red>',sub_item, '</font>'];
      end
      if jj == Sequences{ii}.ActiveTransformation
        sub_item = ['&gt;',sub_item];
        active_transformation_title  = Sequences{ii}.Sequence{jj}.Name;
      else
        sub_item = ['&nbsp;',sub_item];
      end
      ListBoxIndex{end+1}.ImageIndex = handles.Constants.Transformation;
      ListBoxIndex{end}.ProxyIndex = -1; %% arbuz_GetTransIdxByName(handles, Sequences{ii}.Sequence{jj}.Name);
      str{end+1} = ['<html>&nbsp;&nbsp;&nbsp;',sub_item,'</html>'];
    end
  else
    str{end+1} = ['<html>&nbsp;&nbsp;',item_str,'</html>'];
  end
end

% compose the Transformation list
Transformations = arbuz_get(handles.hGUI, 'Transformations');
ActiveTransformation = arbuz_get(handles.hGUI, 'ActiveTransformation');
ListBoxIndex{end+1}.ImageIndex = -1;
str{end+1} = '<html><font color=blue><b>Transformations</b></font></html>';
for ii=1:length(Transformations)
  ListBoxIndex{end+1}.ImageIndex = handles.Constants.Transformation;
  ListBoxIndex{end}.ProxyIndex = ii;
  item_str = Transformations{ii}.Name;
  if strcmp( Transformations{ii}.Name, active_transformation_title)
    str{end+1} = ['<html>&nbsp;&gt;',item_str,'</html>'];
    ActiveTransformation = ii;
  else
    str{end+1} = ['<html>&nbsp;&nbsp;',item_str,'</html>'];
  end
end

val = max(get(handles.lbObjects, 'Value'), 1);
set(handles.lbObjects, 'String', str, 'Value', min(val, length(str)));

% Set Name of the shell
set(handles.hGUI, 'Name', sprintf('ArbuzGUI v2.0: %s [Sequence: %s, Transform.: %s]', ...
  epr_ShortFileName(arbuz_get(handles.hGUI, 'FileName'), 32), ...
  active_sequence_title, active_transformation_title))

handles.ListBoxIndex = ListBoxIndex;
guidata(handles.hGUI, handles);
UpdateReferenceFrameMenu(handles);
UpdateShowSceneMenu(handles);

% --------------------------------------------------------------------
function UpdateReferenceFrameMenu(handles)

str = {'Master frame'};

list_obj = arbuz_FindImage(handles.hGUI, 'all', 'ImageType', '2D', {'FullName'});

for ii=1:length(list_obj)
    str{end+1} = list_obj{ii}.FullName;
end
val = max(get(handles.pmReferenceFrame, 'Value'), 1);
set(handles.pmReferenceFrame, 'String', str, 'Value', min(val, length(str)));

% --------------------------------------------------------------------
function UpdateShowSceneMenu(handles)

str = {'Default';'No output'};

saves = arbuz_get(handles.hGUI, 'saves');
for ii=1:length(saves)
  str{end+1} = [safeget(saves{ii}, 'renderer', '?'), ': ', saves{ii}.name];
end
val = max(get(handles.pmShowGroup, 'Value'), 1);
try
set(handles.pmShowGroup, 'String', str, 'Value', min(val, length(str)));
catch
end

% --------------------------------------------------------------------
% get parameters for image loading
function new_image = GetGUIImageData(handles)

new_image.FileName  = get(handles.eFileName, 'String');
if iscell(new_image.FileName), new_image.FileName = new_image.FileName{1}; end
new_image.Name     = get(handles.eImageTitle, 'String');
if iscell(new_image.Name), new_image.Name = new_image.Name{1}; end
new_image.isStore   = get(handles.cbStoreInProject, 'Value');
new_image.isLoaded  = 0;
new_image.Visible   =  get(handles.cbVisible, 'Value');
new_image.Selected  =  false;
new_image.isStore   =  get(handles.cbStoreInProject, 'Value');

str = get(handles.pmImageType, 'string');
new_image.ImageType = str{get(handles.pmImageType, 'Value')};

% --------------------------------------------------------------------
function OpenProject(fname, handles)

set(handles.hGUI,'Pointer','watch');drawnow
arbuz_OpenProject(handles.hGUI, fname);
set(handles.hGUI,'Pointer','arrow');drawnow

% Update object list index
UpdateListbox(0,handles);
handles = guidata(handles.hGUI);
lbObjects_Callback(handles.lbObjects, [], handles);

seq = arbuz_get(handles.hGUI, 'SEQUENCE');
if ~isempty(seq)
  str = {};
  for jj = 1:length(seq.Sequence)
    str{end+1} = ['Watch after stage: ', seq.Sequence{jj}.Name];
  end
  enable_control = 'on'; pos = seq.WatchTransformation;
  if isempty(str), str = {'Watch: nothing'}; enable_control = 'off'; pos = 1; end
  set(handles.pmSeeStage, 'String', str, 'Value', pos, 'Enable', enable_control);
end

the_list = arbuz_get(handles.hGUI, 'ActionList');
enable_control = 'on';
if isempty(the_list), the_list = {'No actions'}; enable_control = 'off'; end
set(handles.pmRegStage, 'string', the_list, 'Enable', enable_control, 'Value', 1);

% --------------------------------------------------------------------
function SaveProject(fname, handles)

set(handles.hGUI,'Pointer','watch');drawnow
arbuz_ApplyTransformation(handles.hGUI, '', 'fix');
arbuz_SaveProject(handles.hGUI, fname);
set(handles.hGUI,'Pointer','arrow');drawnow

% --------------------------------------------------------------------
% --------------------------------------------------------------------
% ------------- L O A D      I M A G E -------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

function [out, out_pars, imtype] = LoadAutoImage(fname, pars)
set(gcf,'Pointer','watch');drawnow
fprintf('Loading image from ''%s''...\n', fname)
out = [];

[~, ~, fext] = fileparts(fname);
% analyse extension
switch fext
  % ------------.m------------------------------------------------------
  % ---------------.m--------------  TUMOR TOUCH FILE -----------------
  % ------------------.m------------------------------------------------
  % ---------------------.m---------------------------------------------
  case '.m'
    [TumorEllipsoid, isOk] = TumorTouchEditDLG(fname);
    if ~isOk, return; end
    [sel,v] = listdlg('PromptString','Select an option:',...
      'SelectionMode','single','ListSize',[160,160],...
      'ListString', {'3DSURFACE';'MASK'});
    if ~v, return; end
    switch sel
      case 1, out = TumorTouchToSurface(TumorEllipsoid); imtype='3DSURFACE';
      case 2, out = TumorTouchToMask(TumorEllipsoid, pars.Dim, pars.Size/10); imtype='3DMASK';
    end
    pars1.Size = pars.Size;
    pars1.Dim  = pars.Dim;
    pars1.Name = 'Image';
    [res, v] = EditNativeTransformationDLG(pars1);
    if ~v || isempty(res), return; end
    out_pars.Bbox = [1 1 1];
    out_pars.Anative = res.A;
    out_pars.Name = res.Name;
    % -------------.mat----------------------------------------------------
    % -----------------.mat------------ TUMOR TOUCH FILE ------------------
    % ---------------------.mat--------------------------------------------
    % -------------------------.mat----------------------------------------
  case '.mat'
    s = load(fname);
    if isfield(s, 'file_type')
      switch s.file_type
        case 'FitImage_v1.0'
        case 'ImageMask_v1.0'
          [sel,v] = listdlg('PromptString','Select an option:',...
            'SelectionMode','single','ListSize',[160,160],...
            'ListString', {'3DSURFACE';'MASK'});
          if ~v, return; end
          switch sel
            case 1, out = TumorTouchToSurface(s.TumorEllipsoid); imtype='3DSURFACE';
            case 2, out = s.Mask; imtype='3DMASK';
          end
          pars1.Size = pars.Size;
          pars1.Dim  = pars.Dim;
          pars1.Name = 'Image';
          [res, v] = EditNativeTransformationDLG(pars1);
          if ~v || isempty(res), return; end
          out_pars.Bbox = [1 1 1];
          out_pars.Anative = res.A;
          out_pars.Name = res.Name;
      end
    end
end

set(gcf,'Pointer','arrow');drawnow
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------
% --------------------------------------------------------------------

% --------------------------------------------------------------------
function [varargout] = LoadImageParameters(fname, ftype)

set(gcf,'Pointer','watch');drawnow
fprintf('Loading parameters from image from ''%s''...\n', fname)
switch ftype
  case '2D'
    image_info = imfinfo(fname);
    im_pars = [];
    im_box = [image_info.Height, image_info.Width, 0];
  case '3DEPRI'
    [~, ~, ffext] = fileparts(fname);
    switch ffext
      case {'.IMG', '.img'}
      case {'.mat', '.MAT'}
        s = load(fname);
        im_pars = s.imtype;
        im_box = size(s);
    end
  otherwise, varargout{1} = [];
end
disp('Done.');
set(gcf,'Pointer','arrow');drawnow

switch nargout
  case 1, varargout{1} = im_box; % [x1,x2,y1,y2,z1,z2]
  case 2, varargout{1} = im_box; % [x1,x2,y1,y2,z1,z2]
    varargout{2} = im_pars;
end

% --------------------------------------------------------------------
function mFileExportProjectWorkspace_Callback(hObject, eventdata, handles)

prj = getappdata(handles.hGUI, 'project');
assignin('base', 'prj', prj);

disp('Project has been exported to the workspace.')
disp('Variable ''prj'' is created.')

% --------------------------------------------------------------------
function mFileImportProjectWorkspace_Callback(hObject, eventdata, handles)
a = evalin('base', 'exist(''prj'',''var'')');
if a 
  prj = evalin('base', 'prj');
  setappdata(handles.hGUI, 'project', prj);
  
  % Update object list index
  UpdateListbox(0,handles);
  handles = guidata(handles.hGUI);
  lbObjects_Callback(handles.lbObjects, [], handles);
  
  seq = arbuz_get(handles.hGUI, 'SEQUENCE');
  if ~isempty(seq)
    str = {};
    for jj = 1:length(seq.Sequence)
      str{end+1} = ['Watch after stage: ', seq.Sequence{jj}.Name];
    end
    enable_control = 'on'; pos = seq.WatchTransformation;
    if isempty(str), str = {'Watch: nothing'}; enable_control = 'off'; pos = 1; end
    set(handles.pmSeeStage, 'String', str, 'Value', pos, 'Enable', enable_control);
  end
  
  the_list = arbuz_get(handles.hGUI, 'ActionList');
  enable_control = 'on';
  if isempty(the_list), the_list = {'No actions'}; enable_control = 'off'; end
  set(handles.pmRegStage, 'string', the_list, 'Enable', enable_control, 'Value', 1);
  
  arbuz_ShowMessage(handles.hGUI, 'Project has been imported from the workspace (prj structure).');
else
  arbuz_ShowMessage(handles.hGUI, 'Did not find the project.');
end

% --------------------------------------------------------------------
function mSpecialMRIvsEPR_Callback(hObject, eventdata, handles) 

Sequences = arbuz_get(handles.hGUI, 'Sequences');
arbuz_ShowMessage(handles.hGUI, 'Checking the sequence:');
if isempty(Sequences)
  arbuz_ShowMessage(handles.hGUI, 'Create transformations.');
  arbuz_AddTransformation(handles.hGUI, 'T1');
  arbuz_AddTransformation(handles.hGUI, 'M1');
  arbuz_AddTransformation(handles.hGUI, 'T-PET');
  arbuz_AddTransformation(handles.hGUI, 'T-CT');
  arbuz_AddTransformation(handles.hGUI, 'T2');
  arbuz_AddSequence(handles.hGUI, 'S1');

  arbuz_ShowMessage(handles.hGUI, '   new sequence will be created');
  arbuz_ShowMessage(handles.hGUI, 'Setting T1 transformation to the image native.');
  output_list = arbuz_FindImage(handles.hGUI, 'master', '', '', {'Name', 'Anative'});

  for ii=1:length(output_list)
    arbuz_SetTransformation(handles.hGUI, 'T1', output_list{ii}.Name, output_list{ii}.Anative);
    arbuz_ShowMessage(handles.hGUI, sprintf('  image %s was set.', output_list{ii}.Name));
  end
  
  % unselect all images
  arbuz_SetImage(handles.hGUI, output_list, 'Selected', 0);
  
  arbuz_ShowMessage(handles.hGUI, 'Looking for EPR images...');
  find_listEPR = arbuz_FindImage(handles.hGUI, output_list, 'ImageType', '3DEPRI', {'Anative', 'Box'});
  EPR_idx = zeros(length(find_listEPR), 1);
  if ~isempty(find_listEPR)
    for ii=1:length(find_listEPR)
      arbuz_ShowMessage(handles.hGUI, ['  image ', find_listEPR{ii}.Image, ' is found.']);
      EPR_idx(ii) = find_listEPR{ii}.ImageIdx;
    end
  end
  
  arbuz_ShowMessage(handles.hGUI, 'Looking for MRI images...');
  find_listMRI = arbuz_FindImage(handles.hGUI, output_list, 'ImageType', 'MRI', {'Name', 'Anative', 'Box'});
  find_listMRICEST = arbuz_FindImage(handles.hGUI, output_list, 'ImageType', 'CEST_ML', {'Name', 'Anative', 'Box'});
  for jj=1:length(find_listMRICEST), find_listMRI{end+1}=find_listMRICEST{jj}; end
  
  MRI_idx = zeros(length(find_listMRI), 1);
  if ~isempty(find_listMRI)
    MRI_idx = length(find_listMRI);
    
    Aflip = hmatrix_rotate_x(90)*hmatrix_rotate_y(90)*hmatrix_rotate_z(-90);
    for ii=1:length(find_listMRI)
      arbuz_ShowMessage(handles.hGUI, ['  image ', find_listMRI{ii}.Image, ' is found.']);
      arbuz_SetTransformation(handles.hGUI, 'T2', find_listMRI{ii}.Name, Aflip);
      MRI_idx(ii) = find_listMRI{ii}.ImageIdx;
    end
  end
  
  arbuz_ShowMessage(handles.hGUI, 'Looking for Amira images...');
  find_list = arbuz_FindImage(handles.hGUI, output_list, 'ImageType', 'AMIRA3D', {});
  if ~isempty(find_list)
    for ii=1:length(find_list)
      arbuz_ShowMessage(handles.hGUI, ['  found: ',find_list{ii}.Image,'.']);
    end
    arbuz_ShowMessage(handles.hGUI, 'Looking for the reference image ...');
    amira_reference = -1;
    % looking for MRI image
    if ~isempty(find_listMRI)
      amira_reference = find_listMRI{1}.ImageIdx;
      % if not found looking for CT image
    else
      find_listCT = arbuz_FindImage(handles.hGUI, output_list, 'ImageType', 'CT', {'Name', 'Anative', 'Box'});
      if ~isempty(find_listCT)
        amira_reference = find_listCT{1}.ImageIdx;
      end
    end
    
    if amira_reference ~= -1
      str = add_line_to_list(hh, str, ['  found: ',find_list{amira_reference}.Image,'.']);
      idx = find_list{amira_reference}.ImageIdx;
      Aref = handles.images{idx}.data_info.pars_out.world_to_reference;
      for ii=1:length(find_list)
        idx = find_list{ii}.ImageIdx;
        if ii == amira_reference
          if MRI_idx >= 0
            handles.images{find_list{ii}.ImageIdx}.Link = handles.images{MRI_idx}.Name;
          elseif EPR_idx >= 0
            handles.images{find_list{ii}.ImageIdx}.Link = handles.images{EPR_idx}.Name;
          end
          str = add_line_to_list(hh, str, ['T-CT transformation is set for: ',find_list{ii}.Image,'.']);
          handles.images{find_list{ii}.ImageIdx}.LinkTransformation = TCT;
        else
          str = add_line_to_list(hh, str, ['T-PET transformation is set for: ',find_list{ii}.Image,'.']);
          AAref = handles.images{idx}.data_info.pars_out.world_to_reference;
          handles.Transformations{TPET}.Matrices{ii}.A = AAref/Aref; % AAref * inv(Aref)
          handles.Transformations{TPET}.Matrices{ii}.Image = find_list{ii}.Image;
          handles.images{find_list{ii}.ImageIdx}.Link = find_list{amira_reference}.Image;
          handles.images{find_list{ii}.ImageIdx}.LinkTransformation = TPET;
        end
      end
    end
  end
  
  % create sequence
  arbuz_SetActiveSequence(handles.hGUI, 1);
  Sequences = arbuz_get(handles.hGUI, 'Sequences');
  
  arbuz_ShowMessage(handles.hGUI, 'Creating pixel to world (T1) transformation.');
  Sequences{1}.Sequence{1}.Name = 'T1';
  Sequences{1}.Sequence{1}.Description = 'Pixel -> world';

  arbuz_ShowMessage(handles.hGUI, 'Creating MRI mirror (M1) transformation.');
  Sequences{1}.Sequence{2}.Name = 'M1';
  Sequences{1}.Sequence{2}.Description = 'Mirror MRI';
  
  arbuz_ShowMessage(handles.hGUI, 'Creating PET->CT transformation.');
  Sequences{1}.Sequence{3}.Name = 'T-PET';
  Sequences{1}.Sequence{3}.Description = 'PET->CT';
  
  arbuz_ShowMessage(handles.hGUI, 'Creating CT->MRI transformation.');
  Sequences{1}.Sequence{4}.Name = 'T-CT';
  Sequences{1}.Sequence{4}.Description = 'CT->MRI';

  arbuz_ShowMessage(handles.hGUI, 'Creating MRI to EPRI.');
  Sequences{1}.Sequence{5}.Name = 'T2';
  Sequences{1}.Sequence{5}.Description = 'MRI->EPRI'; 
  
  arbuz_set(handles.hGUI, 'Sequences', Sequences);
  arbuz_set(handles.hGUI, 'ACTIVETRANSFORMATION', 'T2');
  arbuz_set(handles.hGUI, 'WATCHTRANSFORMATION', 'T2');

  arbuz_ShowMessage(handles.hGUI, 'Show 3D surfaces.');
  iEPR_color = 0;
  iMRI_color = 0;
  EPR_color{1} = struct('FaceColor', 'none', 'EdgeColor', [0 1 1]);
  EPR_color{2} = struct('FaceColor', 'none', 'EdgeColor', [1 0 1]);
  EPR_color{3} = struct('FaceColor', 'none', 'EdgeColor', [1 0 0]);
  EPR_color{4} = struct('FaceColor', 'none', 'EdgeColor', [1 1 0]);
  EPR_color{5} = struct('FaceColor', 'none', 'EdgeColor', [0 0 1]);
  EPR_color{6} = struct('FaceColor', 'none', 'EdgeColor', [0 1 0]);
  MRI_color{1} = struct('FaceColor', 'none', 'EdgeColor', [0 0 1]*0.75);
  MRI_color{2} = struct('FaceColor', 'none', 'EdgeColor', [0 1 0]*0.75);
  MRI_color{3} = struct('FaceColor', 'none', 'EdgeColor', [1 0 0]*0.75);
  MRI_color{4} = struct('FaceColor', 'none', 'EdgeColor', [1 0 1]*0.75);
  MRI_color{5} = struct('FaceColor', 'none', 'EdgeColor', [0 1 1]*0.75);
  MRI_color{6} = struct('FaceColor', 'none', 'EdgeColor', [1 1 0]*0.75);
  
  find_list = arbuz_FindImage(handles.hGUI, 'all', 'ImageType', '3DSURFACE', {'MasterType'});
  
  for ii=1:length(find_list)
    switch find_list{ii}.MasterType
      case 'MRI'
        the_color = MRI_color{1+mod(iMRI_color, 6)}; iMRI_color = iMRI_color + 1;
      otherwise
        the_color = EPR_color{1+mod(iEPR_color, 6)}; iEPR_color = iEPR_color + 1;
    end
    arbuz_SetImage(handles.hGUI, find_list{ii}, 'SelectedColor', the_color);
    arbuz_SetImage(handles.hGUI, find_list{ii}, 'NotSelectedColor', the_color);
    arbuz_SetImage(handles.hGUI, find_list{ii}, 'Visible', 1);
  end
  
  arbuz_ShowMessage(handles.hGUI, 'MRI-EPR script has finished.');
  UpdateListbox(0,handles);
else
  arbuz_ShowMessage(handles.hGUI, 'Project has Sequence. Script was terminated.');
end

% --------------------------------------------------------------------
function mSpecialMRIsag_Callback(hObject, eventdata, handles)
h = dialog('WindowStyle', 'normal', 'Name', 'Log');
hh = uicontrol(h,'Style', 'edit', 'Tag', 'eList', 'Max', 2, ...
  'Units','normalized', 'Position', [.1 .1 .8 .8], ...
  'HorizontalAlignment', 'left');
T1str = 'T1';
M1str = 'M1';
T2str = 'T2';

arbuz_ShowMessage(handles.hGUI, 'Checking the sequence:');
if length(handles.Sequences) >= 1 && length(handles.Sequences{1}.Sequence) == 3
  arbuz_ShowMessage(handles.hGUI, 'Three transformations found.');
  
  arbuz_ShowMessage(handles.hGUI, 'Looking for ''MRI'' image ...');
  MRI_is = [];
  for ii=1:length(handles.images)
    if strcmp(handles.images{ii}.Name, 'MRI'), MRI_is=ii; break; end
  end
  if isempty(MRI_is), return; end
  arbuz_ShowMessage(handles.hGUI, 'Looking for ''MRI'' image ... found.');
  
  add_line_to_list(hh, str, 'Current image is ...');
  find_list = arbuz_FindImage(handles, 'master', 'Highlighted', 1, {'Name', 'Anative', 'Box'});
  if isempty(find_list), add_line_to_list(hh, str, 'Current image is not of MRI type.'); return; end
  arbuz_ShowMessage(handles.hGUI, sprintf('Current image is ... ''%s''.', find_list{1}.Image));
  if ~strcmp(find_list{1}.ImageType, 'MRI'), return; end
  
  new_matrix.Image = find_list{1}.Name;
  arbuz_ShowMessage(handles.hGUI, 'Setting T1 transformation to the image native.');
  new_matrix.A = find_list{1}.Anative;
  handles = arbuz_AddMatrixToTransformationList(handles, T1str, new_matrix);
  
  arbuz_ShowMessage(handles.hGUI, 'Setting M1 transformation to 90 degree.');
  new_matrix.A = hmatrix_rotate_y(90)*hmatrix_scale([-1, 1, 1]) ;
  handles = arbuz_AddMatrixToTransformationList(handles, M1str, new_matrix);
  
  arbuz_ShowMessage(handles.hGUI, 'Retrieving T2 transformation from MRI image.');
  new_matrix.A = arbuz_GetMatrixByName(handles, T2str, 'MRI');
  handles = arbuz_AddMatrixToTransformationList(handles, T2str, new_matrix);
  
  arbuz_ShowMessage(handles.hGUI, 'Updating transformations.');
  arbuz_SetActiveTransformation(handles.hGUI);
end

% --------------------------------------------------------------------
function mTransformationDeleteAll_Callback(hObject, eventdata, handles) 
arbuz_set(handles.hGUI, 'Transformations', {});
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function mSpecialSetNative_Callback(hObject, eventdata, handles)
% Save active transformation
mTransformationSave_Callback(hObject, eventdata, handles)
handles = guidata(hObject);

find_list = arbuz_FindImage(handles.hGUI, 'master', 'Highlighted', 1, {'Name', 'Anative', 'Box'});
if isempty(find_list), arbuz_ShowMessage(handles.hGUI, 'Current image is not a master image.'); return; end
arbuz_ShowMessage(handles.hGUI, sprintf('Current image is ''%s''.', find_list{1}.Image));

new_matrix.Image = find_list{1}.Name;
arbuz_ShowMessage(handles.hGUI, 'Setting T1 transformation to the image native.');
new_matrix.A = find_list{1}.Anative;
arbuz_SetTransformation(handles.hGUI, 'T1', find_list{1}.Name, new_matrix.A);
arbuz_ShowMessage(handles.hGUI, 'Done.');

% --------------------------------------------------------------------
function ImageContextMenu_Callback(hObject, eventdata, handles) 

delete(get(hObject, 'Children'));

pos = get(handles.lbObjects, 'Value');
idx = handles.ListBoxIndex{pos};

if idx.ImageIndex > 0 && idx.ProxyIndex == -1
    uimenu(hObject, 'Label', 'Add slave image', ...
      'Callback', 'ArbuzGUI(''mImageLoadProxy_Callback'',gcbo,[],guidata(gcbo))');
    uimenu(hObject, 'Label', 'Link to other image', ...
      'Callback', 'ArbuzGUI(''mImageLinkToImage_Callback'',gcbo,[],guidata(gcbo))');

    uimenu(hObject, 'Label', 'Copy all image transformation', ...
      'Callback', 'ArbuzGUI(''mImageCopyPasteTransformation_Callback'',0,[],guidata(gcbo))',...
      'Separator','on');
    uimenu(hObject, 'Label', 'Apply stored transformation', ...
      'Callback', 'ArbuzGUI(''mImageCopyPasteTransformation_Callback'',1,[],guidata(gcbo))');
end

% --------------------------------------------------------------------
function mImageCopyPasteTransformation_Callback(hObject, eventdata, handles)

switch hObject
  case 0
    find_list = arbuz_FindImage(handles.hGUI, 'master', 'Highlighted', 1, {'AALL'});
    
    if ~isempty(find_list)
      arbuz_ApplyTransformation(handles.hGUI, '', 'fix');
      handles.Clipboard.Type = 'ImageTransformation';
      handles.Clipboard.Buffer = find_list{1}.Aall;
      tts = '';
      for ii=1:length(handles.Clipboard.Buffer)
        tts = [tts, ' ', handles.Clipboard.Buffer{ii}.Transformation];
      end
      disp(['Copy image transformations',tts,'.']);
      guidata(handles.hGUI, handles);
    end
  case 1
    find_list = arbuz_FindImage(handles.hGUI, 'master', 'Highlighted', 1, {'AALL'});
    
    if ~isempty(find_list) && isequal(safeget(handles.Clipboard, 'Type', 'ImageTransformation'),  'ImageTransformation')
      stored_transformation = handles.Clipboard.Buffer;
      for ii=1:length(stored_transformation)
        arbuz_SetTransformation(handles.hGUI, stored_transformation{ii}.Transformation,...
          find_list{1}.Image, stored_transformation{ii}.Matrice);
        fprintf('Apply stored transformation %s.', stored_transformation{ii}.Transformation);
      end
    end
end

% --------------------------------------------------------------------
function mImageLoadProxy_Callback(hObject, eventdata, handles)

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName] = uigetfile(handles.FileTypes,'Open file', old_path, ...
  'MultiSelect', 'on');

if PathName ~= 0
  handles.ini.Directories.SourcePath = PathName;
  if ~iscell(FileName), FileName = {FileName}; end
  
  find_list = arbuz_FindImage(handles.hGUI, 'master', 'Highlighted', 1, {'Anative', 'Box'});
  if isempty(find_list), return; end
  
  sz1 = [find_list{1}.Box 1]*find_list{1}.Anative;
  sz2 = [0 0 0 1]*find_list{1}.Anative;
  pars.Dim = find_list{1}.Box;
  pars.Size = sz1(1:3)-sz2(1:3);
  
  for ii=1:length(FileName)
    new_image = [];
    [new_image.data, pars, imtype] = LoadAutoImage(fullfile(PathName, FileName{ii}), pars);
    new_image.Name = pars.Name;
    new_image.A = pars.Anative;
    new_image.isStore = 1;
    handles = arbuz_AddProxyImage(handles, find_list{1}.ImageIdx, new_image, imtype);
  end
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mImageLinkToImage_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles, 'master', 'Highlighted', 1, {'Anative', 'Box'});
if isempty(find_list), return; end

image_to_be_linked = find_list{1}.ImageIdx;

find_list = arbuz_FindImage(handles, 'master', '', 1, {'Anative', 'Box'});
if isempty(find_list), return; end

str = cell(length(find_list), 1); for ii=1:length(find_list), str{ii} = find_list{ii}.Image; end
[s,isOk] = listdlg('PromptString','Select an image:',...
  'SelectionMode','single', 'ListString',str);
if isOk
  image_to_which_will_be_linked = find_list{s}.Image;
  link_transform = handles.ActiveTransformation;
  
  handles.images{image_to_be_linked}.Link = image_to_which_will_be_linked;
  handles.images{image_to_be_linked}.LinkTransformation = link_transform;
  handles = arbuz_SetActiveTransformation(handles, handles.ActiveSequence);
  
  UpdateListbox(0,handles);
end

% --------------------------------------------------------------------
function mSpecialColorImages_Callback(hObject, eventdata, handles)

h = dialog('WindowStyle', 'normal', 'Name', 'Log');
hh = uicontrol(h,'Style', 'edit', 'Tag', 'eList', 'Max', 2, ...
  'Units','normalized', 'Position', [.1 .1 .8 .8], ...
  'HorizontalAlignment', 'left');

MRI_image_colors = [0.4784, 0.0627, 0.8941; 0.0431, 0.5176, 0.7804; 0.4784, 0.0627, 0.8941; 1 0.6000 0.7843];
MRI_image_outline_idx = 1;
MRI_image_fiducials_idx = 1;

str = add_line_to_list(hh, [], 'Looking for pO2 EPR images...');
find_list = arbuz_FindImage(handles, 'master', 'ImageType', 'PO2_pEPRI', {});
for ii=1:length(find_list)
  NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}, 'NotSelectedColor', []);
  SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}, 'SelectedColor', []);
  NotSelectedColor.Color = [1,1,1]; SelectedColor.Color = [1,1,1];
  NotSelectedColor.Color2 = [0.94 0.46 0.84]; SelectedColor.Color2 = [0.94 0.46 0.84];
  NotSelectedColor.LineWidth = 1.5; SelectedColor.LineWidth = 1.5;
  NotSelectedColor.ContourThreshold = [10 20];  SelectedColor.ContourThreshold = [10 20];
  NotSelectedColor.LowCutOff = 0; SelectedColor.LowCutOff = 0;
  NotSelectedColor.HighCutOff = 80; SelectedColor.HighCutOff = 80;
  NotSelectedColor.CutOffAbsoluteScale = 1;  SelectedColor.CutOffAbsoluteScale = 1;
  NotSelectedColor.ColormapName = 'jet'; SelectedColor.ColormapName = 'jet';
  NotSelectedColor.MaskBelowThreshold = 0; SelectedColor.MaskBelowThreshold = 0;
  NotSelectedColor.ShowColorbar = 1; SelectedColor.ShowColorbar = 1;
  NotSelectedColor.SliceErode = 2;  SelectedColor.SliceErode = 2;
  NotSelectedColor.SliceLargest = 1;  SelectedColor.SliceLargest = 1;
  handles.images{find_list{ii}.ImageIdx}.NotSelectedColor = NotSelectedColor;
  handles.images{find_list{ii}.ImageIdx}.SelectedColor = SelectedColor;
  str = add_line_to_list(hh, str, sprintf('  image %s is updated.', find_list{ii}.Image));
end

str = add_line_to_list(hh, str, 'Looking for FID images...');
find_list = arbuz_FindImage(handles, 'all', 'ImageType', '3DSURFACE', {});
for ii=1:length(find_list)
  master_image_type = handles.images{find_list{ii}.ImageIdx}.ImageType;
  
  if (find_list{ii}.ProxyIdx ~= -1) && (~isempty(strfind(find_list{ii}.Image, 'FID')))
    NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'NotSelectedColor', []);
    SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'SelectedColor', []);
    NotSelectedColor.Color = [0,0,1]; SelectedColor.Color = [0,0,1];
    NotSelectedColor.EdgeColor = [0.8706 0.4902 0]; SelectedColor.EdgeColor = [0.8706 0.4902 0];
    NotSelectedColor.FaceColor = 'none'; SelectedColor.FaceColor = 'none';
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.NotSelectedColor = NotSelectedColor;
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.SelectedColor = SelectedColor;
    str = add_line_to_list(hh, str, sprintf('  image %s (%s) is updated.', find_list{ii}.Proxy, find_list{ii}.Image));
    
  elseif (find_list{ii}.ProxyIdx ~= -1) && (~isempty(strfind(find_list{ii}.Image, 'MRI')))
    NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'NotSelectedColor', []);
    SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'SelectedColor', []);
    if ~isempty(strfind(find_list{ii}.Proxy, 'Outline'))
      NotSelectedColor.EdgeColor =  MRI_image_colors(MRI_image_outline_idx,:); SelectedColor.EdgeColor =  MRI_image_colors(MRI_image_outline_idx,:);
      NotSelectedColor.FaceColor = 'none'; SelectedColor.FaceColor = 'none';
      MRI_image_outline_idx = MRI_image_outline_idx + 1;
    else
      NotSelectedColor.EdgeColor =  MRI_image_colors(MRI_image_fiducials_idx,:); SelectedColor.EdgeColor =  MRI_image_colors(MRI_image_fiducials_idx,:);
      NotSelectedColor.FaceColor = 'none'; SelectedColor.FaceColor = 'none';
      MRI_image_fiducials_idx = MRI_image_fiducials_idx + 1;
    end
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.NotSelectedColor = NotSelectedColor;
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.SelectedColor = SelectedColor;
    str = add_line_to_list(hh, str, sprintf('  image %s (%s) is updated.', find_list{ii}.Proxy, find_list{ii}.Image));
    
  elseif (find_list{ii}.ProxyIdx ~= -1) && (~isempty(strfind(find_list{ii}.Image, 'CT')))
    NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'NotSelectedColor', []);
    SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'SelectedColor', []);
    if ~isempty(strfind(find_list{ii}.Proxy, 'Outline'))
      NotSelectedColor.FaceColor =  [0,0.498,0]; SelectedColor.FaceColor = [0,0.498,0];
      NotSelectedColor.EdgeColor = 'none'; SelectedColor.EdgeColor = 'none';
    else
      NotSelectedColor.EdgeColor =  [0,1,0]; SelectedColor.EdgeColor =  [0,1,0];
      NotSelectedColor.FaceColor = 'none'; SelectedColor.FaceColor = 'none';
    end
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.NotSelectedColor = NotSelectedColor;
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.SelectedColor = SelectedColor;
    str = add_line_to_list(hh, str, sprintf('  image %s (%s) is updated.', find_list{ii}.Proxy, find_list{ii}.Image));
    
  elseif (find_list{ii}.ProxyIdx ~= -1) && ...
      (isequal(master_image_type, '3DEPRI') || isequal(master_image_type, 'AMP_pEPRI'))
    NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'NotSelectedColor', []);
    SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'SelectedColor', []);
    NotSelectedColor.EdgeColor = [0.8706 0.4902 0]; SelectedColor.EdgeColor = [0.8706 0.4902 0];
    NotSelectedColor.FaceColor = 'none'; SelectedColor.FaceColor = 'none';
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.NotSelectedColor = NotSelectedColor;
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.SelectedColor = SelectedColor;
    str = add_line_to_list(hh, str, sprintf('  image %s (%s) is updated.', find_list{ii}.Proxy, find_list{ii}.Image));
  end
end

str = add_line_to_list(hh, str, 'Looking for PET images...');
find_list = arbuz_FindImage(handles, 'master', 'ImageType', 'AMIRA3D', {});
for ii=1:length(find_list)
  if ~isempty(strfind(find_list{ii}.Image, 'PET'))
    NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}, 'NotSelectedColor', []);
    SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}, 'SelectedColor', []);
    NotSelectedColor.Color = [0.502,0.502,0.502]; SelectedColor.Color = [0.502,0.502,0.502];
    NotSelectedColor.Color2 = [0,0,0]; SelectedColor.Color2 = [0,0,0];
    NotSelectedColor.LineWidth = 1.5; SelectedColor.LineWidth = 1.5;
    NotSelectedColor.ContourThreshold = [4 7];  SelectedColor.ContourThreshold = [4 7];
    NotSelectedColor.LowCutOff = 0; SelectedColor.LowCutOff = 0;
    NotSelectedColor.HighCutOff = 8; SelectedColor.HighCutOff = 8;
    NotSelectedColor.CutOffAbsoluteScale = 1;  SelectedColor.CutOffAbsoluteScale = 1;
    NotSelectedColor.ColormapName = 'jet'; SelectedColor.ColormapName = 'jet';
    NotSelectedColor.ShowColorbar = 1; SelectedColor.ShowColorbar = 1;
    NotSelectedColor.SliceErode = 0;  SelectedColor.SliceErode = 0;
    NotSelectedColor.SliceLargest = 0;  SelectedColor.SliceLargest = 0;
    NotSelectedColor.MaskBelowThreshold = 0; SelectedColor.MaskBelowThreshold = 0;
    handles.images{find_list{ii}.ImageIdx}.NotSelectedColor = NotSelectedColor;
    handles.images{find_list{ii}.ImageIdx}.SelectedColor = SelectedColor;
    str = add_line_to_list(hh, str, sprintf('  image %s is updated.', find_list{ii}.Image));
  end
end

str = add_line_to_list(hh, str, 'Looking for Tumor masks...');
find_list = arbuz_FindImage(handles, 'all', 'ImageType', '3DMASK', {});
for ii=1:length(find_list)
  %   master_image_type = handles.images{find_list{ii}.ImageIdx}.ImageType;
  if (find_list{ii}.ProxyIdx ~= -1) && (~isempty(strfind(find_list{ii}.Proxy, 'Tumor')))
    NotSelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'NotSelectedColor', []);
    SelectedColor =  safeget(handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}, 'SelectedColor', []);
    NotSelectedColor.Color = [1,0,0]; SelectedColor.EdgeColor = [1,0,0];
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.NotSelectedColor = NotSelectedColor;
    handles.images{find_list{ii}.ImageIdx}.proxy{find_list{ii}.ProxyIdx}.SelectedColor = SelectedColor;
    str = add_line_to_list(hh, str, sprintf('  image %s (%s) is updated.', find_list{ii}.Proxy, find_list{ii}.Image));
  end
end

add_line_to_list(hh, str, 'Finished.');
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function AddMessage(handles, the_message)
str = get(handles.eLogWindow, 'string');
if ~iscell(str), str = {str}; end
str{end+1} = the_message;
if length(str) > 5
    str = str(length(str)-(5-1):end);
end
set(handles.eLogWindow, 'string', str);

% --------------------------------------------------------------------
function pmSeeStage_Callback(hObject, eventdata, handles)
val = get(handles.pmSeeStage, 'Value');
arbuz_set(handles.hGUI, 'WatchTransformation', val);
UpdateListbox(0,handles);

% --------------------------------------------------------------------
function pmRegStage_Callback(hObject, eventdata, handles)
val = get(handles.pmRegStage, 'Value');
Actions = arbuz_get(handles.hGUI, 'Actions');

t_name = arbuz_get(handles.hGUI, 'ActiveTransformationName');
arbuz_StageToTransformation(handles.hGUI, t_name);
arbuz_set(handles.hGUI, 'ActiveTransformation', Actions{val}.Stage);

arbuz_SetImage(handles.hGUI, 'master', 'selected', 0);
for ii=1:length(Actions{val}.Images)
  arbuz_SetImage(handles.hGUI, Actions{val}.Images(ii), 'selected', 1);
end

UpdateListbox(0,handles);


% --------------------------------------------------------------------
function mImageLimitData_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'data'});
if isempty(find_list), return; end

data_min = min(find_list{1}.data(:));
data_max = max(find_list{1}.data(:));
res = inputdlg({'Data mininmum', 'Data maximum'}, 'Data min/max', 1, {num2str(data_min),num2str(data_max)});

if ~isempty(res)
  data_min = str2double(res{1});
  data_max = str2double(res{2});
  data = find_list{1}.data;
  data(find_list{1}.data >= data_max) = data_max;
  data(find_list{1}.data <= data_min) = data_min;
  arbuz_SetImage(handles.hGUI, find_list(1), 'data', data);
end


% --------------------------------------------------------------------
function mActionAdd_Callback(hObject, eventdata, handles)
find_list = arbuz_FindImage(handles.hGUI, 'all', 'Highlighted', 1, {'data'});
Stages = arbuz_get(handles.hGUI, 'Stages');

[s,ok] = listdlg('PromptString','Select a stage:',...
'SelectionMode','single','ListString',Stages);

if ok
  action.Stage  = s;
  action.Images = {};
  action.Description = ['Stage ', num2str(s), ' for '];
  for ii=1:length(find_list)
    action.Images{end+1} = find_list{ii}.Image;
    action.Description = [action.Description, find_list{ii}.Image, ', '];
  end
  action.Description = action.Description(1:end-2);
  
  s = inputdlg({'Enter description'}, 'Action', 1, {action.Description});
  if ~isempty(s)
    action.Description = s{1};
    Actions = arbuz_get(handles.hGUI, 'Actions');
    Actions{end+1} = action;
    arbuz_set(handles.hGUI, 'Actions', Actions);
    
    the_list = arbuz_get(handles.hGUI, 'ActionList');
    enable_control = 'on';
    if isempty(the_list), the_list = {'No actions'}; enable_control = 'off'; end
    set(handles.pmRegStage, 'string', the_list, 'Enable', enable_control, 'value', min(handles.pmRegStage.Value, length(the_list)));
  end
end

% --------------------------------------------------------------------
function mActionDelete_Callback(hObject, eventdata, handles)
val = get(handles.pmRegStage, 'value');

Actions = arbuz_get(handles.hGUI, 'Actions');
Actions = Actions([1:val-1, val+1:end]);
arbuz_set(handles.hGUI, 'Actions', Actions);

the_list = arbuz_get(handles.hGUI, 'ActionList');
enable_control = 'on';
if isempty(the_list), the_list = {'No actions'}; enable_control = 'off'; end
set(handles.pmRegStage, 'string', the_list, 'value', length(the_list), 'Enable', enable_control);

% --------------------------------------------------------------------
function pbImSel_Callback(hObject, eventdata, handles)

UpdateListbox(0, handles);

% --------------------------------------------------------------------
function MainFigure_CloseRequestFcn(hObject, eventdata, handles)

res = arbuz_get(handles.MainFigure,'canclose');

if res
  delete(hObject);
else
  ButtonName = questdlg('The project is not saved. Do you wand to close it ?', ...
                        'Closing Dialog', ...
                         'Yes', 'No', 'No');  
  switch ButtonName
    case 'Yes', delete(hObject);
  end
end


% --------------------------------------------------------------------
function Export_as_slaves_Callback(hObject, eventdata, handles)
% hObject    handle to Export_as_slaves (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
pos = get(handles.lbObjects, 'Value');

Output_fields = {'Name','DATA', 'Filename','Ashow','bbox','Anative'};
highlighted = [];
hh = arbuz_FindGUI();
prj = getappdata(hh, 'project');
[Path, Project]= fileparts(prj.FileName);
status = mkdir(Path,'Export') ;
disp('Creating Directory for export')
Highlighted_image = arbuz_FindImage(hh,'master','HIGHLIGHTED',[],Output_fields);
Master_images = arbuz_FindImage(hh,'master','',[],Output_fields);
   
All = tic();
for ii=1:length(Master_images)   
    tic
   if length(Highlighted_image)>1; disp('More than one image selected ... exiting'); continue; end
   if isempty(Highlighted_image); disp('No image selected for export frame ... exiting'); continue; end
%    if ~(Highlighted_image{1}.ImageIdx == Master_images{ii}.ImageIdx)
       disp(sprintf('%s','Creating Directory for ',Master_images{ii}.Name));
       status = mkdir(sprintf('%s',Path,filesep,'Export'),Master_images{ii}.Name);       
       disp(sprintf('%s','Transforming ',Master_images{ii}.Name,' Into corrdinates of ',Highlighted_image{1}.Name));     
       resliced_volume = reslice_volume(inv(Highlighted_image{1}.Ashow), inv(Master_images{ii}.Ashow), zeros(size(Highlighted_image{1}.data)), double(Master_images{ii}.data),0, 1);
       fname = sprintf('%s',Path,filesep,'Export',filesep,Master_images{ii}.Name,filesep,Master_images{ii}.Name,'.img');
       fprintf('Saving image as %s\n',fname);
            
            dd = diag(Highlighted_image{1}.Anative);
            sz = Highlighted_image{1}.Box(:).*dd([2,1,3]);
            options.BoundingBox = [-sz(1:3)/2, sz(1:3)/2]';
            options.BoundingBox = options.BoundingBox(:)';
            write_binary_image_file(fname, permute(resliced_volume,[2,1,3]), options)
       
%    end 
   toc
   
end
fprintf('Images Exported in the frame of %s',  Highlighted_image{1}.Name);
fprintf('Images exported to %s\n',fileparts(fname)) ;

toc(All);

% --------------------------------------------------------------------
function Chuck_depth_vs_Dose_Callback(hObject, ~, handles)
% hObject    handle to Chuck_depth_vs_Dose (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

Output_fields = {'Name','DATA', 'Filename','Ashow','bbox','Anative'};
hh = arbuz_FindGUI();
prj = getappdata(hh, 'project');

res = arbuz_FindImage(hh, 'master', 'ImageType', 'DICOM3D',Output_fields);

if length(res)==1
CT = res{1}.data;
else
    disp('More or Less than 1 CT found')
    return
end


slnum = size(CT,3);
if min(CT(:)) < -100, CT=(double(CT)+1000.0)/5000.0;end

Thresholds=FindskinGUI(CT);

midslice=squeeze(CT(:,:,floor(slnum/2)));
figure(200),subplot(1,2,1),imagesc(midslice), axis image,colormap(gca, gray);
pause(0.1);

skinmask = CT*0;
skinmask(find(CT<Thresholds.Skin_high & CT>Thresholds.Skin_low))=1;
% skinmask = imerode(imdilate(skinmask,se),se);
pvmask = CT*0;
pvmask(find(CT<Thresholds.PVS_high & CT>Thresholds.PVS_low))=1;
ringmask = CT*0;
ringmask(find(CT<Thresholds.Ring_high & CT>Thresholds.Ring_low))=1;

totmask = CT*0;
totmask(find(skinmask))=1;
totmask(find(pvmask))=3;
totmask(find(ringmask)) =2;

material_atten=[0.0206 0.0205 0.029];  % attenuation 1/mm for materials
calibration_atten=0.019;    % measured attenuation for solid water at 225 kVp

midcol=floor(size(midslice,2)/2);
vertprof=midslice(:,midcol);
midmask=squeeze(totmask(:,:,floor(slnum/2)));
subplot(1,2,2),imagesc(midmask),colormap(gca, jet),axis image
midline=midmask(:,midcol);

middens=0*midline;
for i=1:3
middens(midline==i)=material_atten(i);
end
pixsiz=0.1;
midpath=middens*pixsiz/calibration_atten;  % radiological depth per pixel
uppath=[];
downpath=[];
where=[];

% limits for evaluating 
icen=round(size(midslice,1)/2);
ilo=icen-100;
ihi=icen+40;

for i=ilo:ihi
where(end+1)=i;
uppath(end+1)=sum(midpath(1:i));
downpath(end+1)=sum(midpath(i+1:end));
end
where=pixsiz*(where-icen);

downlsq=ones(size(where))*307^2./(307-where)./(307-where);
uplsq=ones(size(where))*307^2./(307+where)./(307+where);

dosedepths=[0	0.2	0.4	0.9	1.4	1.9	2.4	3.4  4.4  5.4];
dosedepths=dosedepths*10;
colsizes=[3.5 2.5 1.85 1.5 0.75 0.5];
mysize=3.5;
mycol=find(colsizes==mysize);

dosetable=[2.8573    2.8528    2.8156    2.7195    2.5826    2.3633
2.8627    2.8074    2.7721    2.6452    2.5210    2.2472
2.7594    2.6760    2.6443    2.5718    2.4358    2.1475
2.4992    2.4738    2.4476    2.2953    2.1203    1.8792
2.2771    2.2409    2.1112    2.0143    1.9209    1.6607
2.0641    1.9626    1.8982    1.8157    1.6870    1.4205
1.8583    1.7758    1.6643    1.5782    1.4404    1.2591
1.5066    1.3706    1.3244    1.2673    1.0887    0.9591
1.1068    1.1884    1.0135    0.9890    0.8639         0
0.9092    0.9663    0.7578    0.7878    0.6772         0];

mycoldose=dosetable(:,mycol);

updose=uppath*0;
downdose=downpath*0;
for i=1:numel(uppath)
    updose(i)=interp1(dosedepths,mycoldose,uppath(i));
    downdose(i)=interp1(dosedepths,mycoldose,downpath(i));
end
updose=updose.*uplsq;
downdose=downdose.*downlsq;
totdose=updose+downdose;

figure(201)
subplot(1,2,1);
plot(where,updose)
hold on
plot(where,downdose)
legend('dose from above','dose from below')
xlabel('mm from image center')
subplot(1,2,2);
plot(where,totdose);
legend('total');
xlabel('mm from image center')
ylim=get(gca,'ylim');
ylim(1)=0.8*ylim(2);
set(gca,'ylim',ylim);

% --- Executes on button press in pbIbGUI.
function pbIbGUI_Callback(hObject, eventdata, handles)
ibGUI(handles.eFileName.String)
