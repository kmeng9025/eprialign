function varargout = ProcessGUI(varargin)
% PROCESSGUI M-file for ProcessGUI.fig
%      PROCESSGUI, by itself, creates a new PROCESSGUI or raises the
%      existing
%      singleton*.
%
%      H = PROCESSGUI returns the handle to a new PROCESSGUI or the handle to
%      the existing singleton*.
%
%      PROCESSGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROCESSGUI.M with the given input arguments.
%
%      PROCESSGUI('Property','Value',...) creates a new PROCESSGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before ProcessGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to ProcessGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help ProcessGUI

% Last Modified by GUIDE v2.5 19-Jun-2023 17:18:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @ProcessGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @ProcessGUI_OutputFcn, ...
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
function ProcessGUI_OpeningFcn(hObject, eventdata, handles, varargin)

handles.ini = inimanage(epr_GetIniPath('ProcessGUI'));
handles.ini.Directories = safeget(handles.ini, 'Directories', []);
handles.ini.Directories.Output = safeget(handles.ini.Directories, 'Output', []);
handles.output = hObject;
handles.Groups = {};

handles.fields = []; % fields ready to use in processing routines
handles.pars   = []; % fields as read from ini file

Scenario = safeget(handles.ini, 'Scenario', []);
ScenarioLast = safeget(Scenario, 'Last', '');
Parameters = safeget(handles.ini, 'Parameters', []);
ParametersLast = safeget(Parameters, 'Last', '');

if exist(ScenarioLast, 'file')
  LoadScenario(ScenarioLast, handles);
  if exist(ParametersLast, 'file')
    LoadParameters(ParametersLast, guidata(hObject));
  end
else
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function varargout = ProcessGUI_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --------------------------------------------------------------------
function figure1_CloseRequestFcn(hObject, eventdata, handles)
try
  if isfield(handles, 'ini')
    inimanage(epr_GetIniPath('ProcessGUI'), handles.ini);
  end
catch
end
delete(hObject);

% --------------------------------------------------------------------
function figure1_ResizeFcn(hObject, eventdata, handles)
ps = get(handles.figure1, 'Position');

% enlarge font size when increase window size
font_size = fix(8 + (sqrt(ps(3)^2+ps(4)^2) - 450)*0.009);
hh = [handles.tblParameters, handles.pmValue, handles.eValue, handles.pmParGroups];
set(hh, 'FontSize', font_size);

border = 12;
upper_controls  = 35;
bottom_controls = 70;
middle_controls1 = 30;
middle_controls2 = 25;

x1 = border; x2 = ps(3)/2-border/2; x3 = ps(3)/2+border/2; x4 = ps(3)-border;
y1 = border; y2 = bottom_controls+border; y3 = ps(4) - upper_controls - border; y4 = ps(4)-border;

set(handles.tblParameters, 'Position', [x1,y2,x2-x1,y3-y2]);
set(handles.tblParameters, 'ColumnWidth', {(x2-x1-20)*.55,(x2-x1-20)*.45})
set(handles.pmParGroups, 'Position', [x1,y3,x2-x1,y4-y3]);
set(handles.uipanelvalue, 'Position', [x1,y1,x2-x1,y2-y1]);

y5 = middle_controls1+bottom_controls+border;
y6 = middle_controls1+middle_controls2+bottom_controls+border;

set(handles.text1, 'Position', [x3,y3+10,50,15]);
set(handles.cbClearList, 'Position', [x3+30,y3,x4-x3-50,y4-y3]);
set(handles.cbAverage, 'Position', [x3+180,y3,x4-x3-50,y4-y3]);
set(handles.eOutputDirectory, 'Position', [x3,y2,x4-x3-40,y5-y2]);
set(handles.pbOutputDir, 'Position', [x4-40,y2,40,y5-y2]);
set(handles.lbData, 'Position', [x3,y6,x4-x3,y3-y6]);
set(handles.pmViewerSelector, 'Position', [x3,y5,90,y6-y5]);
set(handles.pbShow, 'Position', [x3+95,y5,55,y6-y5]);
set(handles.eFileSuffix, 'Position', [x4-70,y5,70,y6-y5]);
set(handles.pbProcess, 'Position', [x3,y1,x4-x3,y2-y1]);

% --------------------------------------------------------------------
function pmParGroups_Callback(hObject, eventdata, handles)
if isempty(handles.Groups), mFileScenario_Callback(hObject, eventdata, handles); return; end

gr = get(handles.pmParGroups, 'Value');
if gr < 1, return; end

pars = handles.Groups{gr}.Parameters;
flds = handles.fields.(handles.Groups{gr}.Field);
vals = handles.pars.(handles.Groups{gr}.Field);
lp = length(pars);

data_tbl = cell(lp,2);
for ii=1:lp
  data_tbl{ii,1}=pars{ii}.Name;
  switch pars{ii}.Type
    case {'IDX', 'IDX0'}
      idx = safeget(flds, pars{ii}.Field, 1);
      data_tbl{ii,2}=pars{ii}.Show{idx};
    otherwise
      data_tbl{ii,2}=safeget(vals, pars{ii}.Field, '');
  end
end

set(handles.tblParameters, 'Data', data_tbl);

% --------------------------------------------------------------------
function lbData_Callback(hObject, ~, handles)

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end

% --------------------------------------------------------------------
function pbProcess_Callback(hObject, eventdata, handles)

% request necessary fields
[handles, isOk] = RequestFields(handles);
if ~isOk, return; end

% get processing parameters
prc = safeget(handles.fields, 'prc', struct('process_method', 'ese_fbp'));
process_method = safeget(prc, 'process_method', 'ese_fbp');
launch_viewer = strcmp(safeget(prc, 'launch_viewer', 'yes'), 'yes');

set(handles.figure1,'Pointer','watch');drawnow

fname =  get(handles.eOutputDirectory, 'String');
[output_path]=fileparts(fname);

files = get(handles.lbData, 'String');
if ~iscell(files), files = {files}; end

file_suffix = get(handles.eFileSuffix,'String');

if get(handles.cbAverage, 'Value') == 0
  for ii=1:length(files)
    disp(['Processing  ', files{ii}]);
    try
      handles.Image = feval(process_method, files{ii}, file_suffix, output_path, handles.fields);
      drawnow;
      
    catch err
      disp(' ');
      disp(['ERROR! Data file ''',files{ii},'''is not processed.']);
      disp(err.message); disp(' ');
      set(handles.lbData, 'BackgroundColor', [1, .92, .92]);
      launch_viewer = 0;
    end
    
    drawnow;
  end
else
  try
    handles.Image = feval(process_method, files, file_suffix, output_path, handles.fields);
  catch err
    disp(' ');
    disp('ERROR! Data file is not processed.');
    disp(err.message); disp(' ');
    set(handles.lbData, 'BackgroundColor', [1, .92, .92]);
    launch_viewer = 0;
  end
end

set(handles.lbData, 'BackgroundColor', [.95, .95, 1]);
set(handles.figure1,'Pointer','arrow');drawnow

guidata(handles.figure1, handles);

if ~isempty(handles.Image)
  disp('Batch processing is finished.');
  
  if launch_viewer
    pbShow_Callback(hObject, eventdata, handles);
  end
else
  disp('Processing failed.');  
end

% --------------------------------------------------------------------
function mFileLoad_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/'); 

[FileName,PathName] = uigetfile( ...
  { '*.d01;*.tdms', 'SpecMan files (*.d01;*.tdms)'; ...
  '*.img', 'LabView image files (*.img)'; ...
  '*.mat', 'Matlab image files (*.mat)'; ...
  '*.tdms', 'TDMS image files (*.tdms)'; ...
  '*.dsc', 'DSC Bruker files (*.dsc)'; ...
  '*.img;*.d01;*.mat;*.tdms;*.dsc', 'All supported data (*.img;*.d01;*.mat;*.tdms;*.dsc)'},  ...
  'Open file', old_path, ...  
  'MultiSelect', 'on');
if PathName ~= 0
  handles.ini.Directories.SourcePath = PathName;
  files = iff(hObject == handles.mFileLoad, {} , get(handles.lbData, 'String'));
  if handles.mFileSetPathInput.Checked
    set(handles.eOutputDirectory, 'String', PathName);
  end
  if iscell(FileName)
    for ii=1:length(FileName)
      files{end+1} = fullfile(PathName, FileName{ii});
    end
  else
    files = {fullfile(PathName, FileName)};
  end
  files = unique(files);
  set(handles.lbData, 'String', files, 'Value', 1);
  
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function mFileScenario_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
fp = fileparts(get(handles.figure1, 'FileName'));
old_path = safeget(dirs, 'Scenario', fullfile(fp, 'Scenario')); 

[FileName,PathName] = uigetfile({'*.scn', 'Scenario files (*.scn)'; '*.*', 'All files (*.*)'},'Open file', old_path, ...
  'MultiSelect', 'off');
if PathName ~= 0
  handles.ini.Directories.Scenario = PathName;
  handles.ini.Scenario.Last = fullfile(PathName, FileName);
  LoadScenario(handles.ini.Scenario.Last, handles);
  
  mFileLoadPars_Callback(hObject, eventdata, guidata(handles.figure1));
end

% --------------------------------------------------------------------
function LoadScenario(fname, handles)

if ~exist(fname, 'file'), return; end

[handles.Groups, info] = ProcessLoadScenario(fname, '');
set(handles.figure1, 'Name', info.Scenario);
handles.Scenario = info.Scenario;

handles.request_fld = info.request_fld;

for ii=1:length(handles.Groups)
    for jj=1:length(handles.Groups{ii}.Parameters)
      switch handles.Groups{ii}.Parameters{jj}.Type
        case {'D', 'S', 'F'}, handles.Groups{ii}.Parameters{jj}.Control = handles.eValue;
        case {'IDXS','IDX','IDX1+'},  handles.Groups{ii}.Parameters{jj}.Control = handles.pmValue;
      end
    end
end

set(handles.pmParGroups, 'String', info.GroupNames, 'Value', 1)

handles.fields = [];
handles = UpdateFields(handles);
if ~isempty(handles.Groups), pmParGroups_Callback(handles.pmParGroups, [], handles); end
guidata(handles.figure1, handles);
disp(sprintf('Scenario is loaded from %s.', fname));

% --------------------------------------------------------------------
function [handles, isOk] = RequestFields(handles)
isOk =true;

for ii=1:length(handles.request_fld)
  Group = safeget(handles.request_fld{ii}, 'Group', []);
  Field = safeget(handles.request_fld{ii}, 'Field', []);
  if ~isempty(Group) && ~isempty(Field)
    Gr  = GetGroupByName(handles.Groups, Group);
    Pr  = GetParameterByName(Gr, Field);
    [tmp_pars, tmp_fields, isOk] = ProcessValueDialogDLG(handles.pars.(Pr.Group), handles.fields.(Pr.Group), Pr, handles.ini);
    if isOk
      handles.pars.(Pr.Group) = tmp_pars;
      handles.fields.(Pr.Group) = tmp_fields;
    else
      return;
    end
  end
end
guidata(handles.figure1, handles);
pmParGroups_Callback(handles.figure1, [], handles);

% --------------------------------------------------------------------
function gr = GetGroupByName(Groups, group_name)
gr = [];
for ii=1:length(Groups)
  if strcmpi(Groups{ii}.Field, group_name)
    gr = Groups{ii}; break;
  end
end

% --------------------------------------------------------------------
function par = GetParameterByName(Group, par_name)
Parameters = safeget(Group, 'Parameters', []);
par = [];
for ii=1:length(Parameters)
  if strcmpi(Parameters{ii}.Field, par_name)
    par = Parameters{ii}; break;
  end
end

% --------------------------------------------------------------------
function mFileLoadPars_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
fp = fileparts(get(handles.figure1, 'FileName'));
old_path = safeget(dirs, 'Parameters', fp); 

[FileName,PathName] = uigetfile({'*.par', 'Parameters files (*.par)'; '*.*', 'All files (*.*)'},'Open file', old_path, ...
  'MultiSelect', 'off');
if PathName ~= 0
  handles.ini.Directories.Parameters = PathName;
  handles.ini.Parameters.Last = fullfile(PathName, FileName);
  LoadParameters(handles.ini.Parameters.Last, handles);
end

% --------------------------------------------------------------------
function LoadParameters(fname, handles)
handles.pars = inimanage(handles.ini.Parameters.Last);
handles = UpdateFields(handles);
guidata(handles.figure1, handles);
pmParGroups_Callback(handles.figure1, [], handles);
fprintf('Parameters are loaded from %s.\n', handles.ini.Parameters.Last);
set(handles.figure1, 'Name', sprintf('%s [%s]', handles.Scenario, handles.ini.Parameters.Last))

% --------------------------------------------------------------------
function handles = UpdateFields(handles)
for gr=1:length(handles.Groups)
  par_dsc = handles.Groups{gr}.Parameters;
  gr_Field = handles.Groups{gr}.Field;
  if ~isfield(handles.fields, gr_Field), handles.fields.(gr_Field)=[]; end
  if ~isfield(handles.pars, gr_Field), handles.pars.(gr_Field)=[]; end
  [handles.fields.(gr_Field), handles.pars.(gr_Field)] = ProcessFormController(handles.pars.(gr_Field), handles.fields.(gr_Field), par_dsc, 'ini2fields');
end

% --------------------------------------------------------------------
function eValue_Callback(hObject, eventdata, handles)
gr = get(handles.pmParGroups, 'Value');
if gr < 1, return; end
par = get(handles.eValue, 'UserData');

[handles.pars.(par.Group), handles.fields.(par.Group)] = ...
  ProcessFormController(handles.pars.(par.Group), handles.fields.(par.Group), par,'gui2ini');
guidata(handles.figure1, handles);
pmParGroups_Callback(hObject, [], handles);

% --------------------------------------------------------------------
function pmValue_Callback(hObject, ~, handles)
gr = get(handles.pmParGroups, 'Value');
if gr < 1, return; end
par = get(handles.pmValue, 'UserData');
[handles.pars.(par.Group), handles.fields.(par.Group)] = ...
  ProcessFormController(handles.pars.(par.Group), handles.fields.(par.Group), par,'gui2ini');
guidata(handles.figure1, handles);
pmParGroups_Callback(hObject, [], handles);

% --------------------------------------------------------------------
function pbValueFSelect_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/'); 

[FileName,PathName] = uigetfile({'*.mat', 'Matlab files (*.mat)'; '*.*', 'All files (*.*)'},'Open file', old_path, ...
  'MultiSelect', 'off');
if PathName ~= 0
  set(handles.eValue, 'String', fullfile(PathName, FileName));
  eValue_Callback(handles.eValue, eventdata, handles)
end

% --------------------------------------------------------------------
function tblParameters_CellSelectionCallback(hObject, eventdata, handles)
if isempty(eventdata.Indices), return; end
gr = get(handles.pmParGroups, 'Value');
if gr < 1, return; end
par = handles.Groups{gr}.Parameters{eventdata.Indices(1)};

set(handles.uipanelvalue, 'Title', sprintf('Value (%s.%s)',par.Group, par.Field))
set([handles.eValue, handles.pmValue, handles.pbValueFSelect], 'Visible', 'off');
switch par.Type
  case {'D', 'S', 'F'}
    set(handles.eValue, 'Visible', 'on', 'UserData', par)
    ProcessFormController(handles.pars.(par.Group), handles.fields.(par.Group), par,'ini2gui');
    if par.Type == 'F'
      set(handles.pbValueFSelect, 'Visible', 'on', 'UserData', par)
    end
  case {'IDX','IDX1+','IDXS'}
    set(handles.pmValue, 'String', par.Show, 'Value', 1, 'UserData', par, 'Visible', 'on')    
    ProcessFormController(handles.pars.(par.Group), handles.fields.(par.Group), par, 'ini2gui');
end
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function mFileSave_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
fp = fileparts(get(handles.figure1, 'FileName'));
old_path = safeget(dirs, 'Parameters', fp); 

[FileName,PathName] = uiputfile({'*.par', 'Parameters files (*.par)'; '*.*', 'All files (*.*)'},'Open file', old_path);
if PathName ~= 0
  handles.ini.Directories.Parameters = PathName;
  handles.ini.Parameters.Last = fullfile(PathName, FileName);
  inimanage(handles.ini.Parameters.Last, handles.pars);
end

% --------------------------------------------------------------------
function mParsShowAll_Callback(hObject, eventdata, handles)
disp(' '); disp('Fields')
for ii=1:length(handles.Groups)
  disp(['  --- ',handles.Groups{ii}.Field]);
  disp(handles.fields.(handles.Groups{ii}.Field))
end
disp(' '); disp('Pars')
for ii=1:length(handles.Groups)
  disp(['  --- ',handles.Groups{ii}.Field]);
  disp(handles.pars.(handles.Groups{ii}.Field))
end

% --------------------------------------------------------------------
function mOptRoot_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
dname = uigetdir(safeget(dirs, 'Output', ''));

if dname~=0
  handles.ini.Directories.Output = dname;
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function pbOutputDir_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/'); 

fpath = uigetdir(old_path, 'Pick a Directory');

set(handles.eOutputDirectory, 'String', [fpath,'\']);


% --------------------------------------------------------------------
function pbShow_Callback(hObject, eventdata, handles)
switch get(handles.pmViewerSelector, 'Value')
  case 1, ibGUI(handles.Image, safeget(handles.Image, 'ImageName', ''))
  case 2, sliceomatic(double(handles.Image.Raw)); cameratoolbar('hide');
  case 3, sliceomatic(double(handles.Image.Amp)); cameratoolbar('hide');
end

% --------------------------------------------------------------------
function mFileExportPars_Callback(hObject, eventdata, handles)
assignin('base', 'processing_struct', handles.fields);
disp('''processing_struct'' is created in the base workspace.');


% --------------------------------------------------------------------
function mFileExportResult_Callback(hObject, eventdata, handles)
assignin('base', 'Image', handles.Image);
disp('''Image'' is created in the base workspace.');
disp(handles.Image);

% --------------------------------------------------------------------
function mOptPath2_Callback(hObject, eventdata, handles)
files = get(handles.lbData, 'String');
if ~iscell(files), files = {files}; end

fpath = [fileparts(files{1}), filesep];
set(handles.eOutputDirectory, 'String', fpath);

% --------------------------------------------------------------------
function mFileTodayLoad_Callback(hObject, eventdata, handles)
dirs     = safeget(handles.ini, 'Directories', []);
old_path = epr_PathFromDate(date, 'pulse250', '\'); 

[FileName,PathName] = uigetfile( ...
  { '*.d01;*.tdms', 'SpecMan files (*.d01);*.tdms'; ...
  '*.img', 'LabView image files (*.img)'; ...
  '*.mat', 'Matlab image files (*.mat)'; ...
  '*.tdms', 'TDMS image files (*.tdms)'; ...
  '*.dsc', 'DSC Bruker files (*.dsc)'; ...
  '*.img;*.d01;*.mat;*.tdms;*.dsc', 'All supported data (*.img;*.d01;*.mat;*.tdms;*.dsc)'},  ...
  'Open file', old_path, ...  
  'MultiSelect', 'on');
if PathName ~= 0
  handles.ini.Directories.SourcePath = PathName;
  files = iff(hObject == handles.mFileLoad, {} , get(handles.lbData, 'String'));
  
  [~,~,ret]=epr_DateFromPath(PathName);
  set(handles.eOutputDirectory, 'String', epr_PathFromDate(ret.datestr, 'imagnet', [ret.folder,'\']));

  
  if iscell(FileName)
    for ii=1:length(FileName)
      files{end+1} = fullfile(PathName, FileName{ii});
    end
  else
    files = {fullfile(PathName, FileName)};
  end
  files = unique(files);
  set(handles.lbData, 'String', files, 'Value', 1);
  
  guidata(handles.figure1, handles);
end

% --------------------------------------------------------------------
function mParsEditScenario_Callback(~, ~, handles)
eval(['edit ' , handles.ini.Scenario.Last]);

% --------------------------------------------------------------------
function mParsEditPar_Callback(~, ~, handles)
eval(['edit ' , handles.ini.Parameters.Last]);

% --------------------------------------------------------------------
function mCalibrationLoad_Callback(hObject, eventdata, handles)
str = epr_spin_probe_calibration();
[s,isOK] = listdlg('PromptString','Select calibration:', 'SelectionMode','single','ListString',str);
if isOK
  spin_probe = epr_spin_probe_calibration(str{s});
  
  if isfield(handles.fields, 'clb')
    disp(spin_probe);
    handles.fields.clb
    method = 'IRESE';
    f = 1/pi/2/2.802*1000; % conversion of MS-1 to mG
    if contains(method, 'IRESE')
% [handles.pars.clb, handles.fields.clb] = ...
%   ProcessFormController(handles.pars.clb, handles.fields.clb, par,'gui2ini');
% [handles.pars.clb, handles.fields.clb] = ...
%   ProcessFormController(handles.pars.clb, handles.fields.clb, par,'gui2ini');
      handles.fields.clb.LLW_zero_po2 = spin_probe.interceptT1 * f;
      handles.fields.clb.Torr_per_mGauss =  spin_probe.slopeT1 / f;
      handles.pars.clb.LLW_zero_po2 = num2str(handles.fields.clb.LLW_zero_po2);
      handles.pars.clb.Torr_per_mGauss = num2str(handles.fields.clb.Torr_per_mGauss);
    else
      handles.fields.clb.LLW_zero_po2 = spin_probe.interceptT2 * f;
      handles.fields.clb.Torr_per_mGauss =  spin_probe.slopeT2 / f;
    end
    guidata(handles.figure1, handles);
    pmParGroups_Callback(handles.figure1,[],handles);
  else
    error('clb field is not found.');
  end
end
