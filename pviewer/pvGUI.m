function varargout = pvGUI(varargin)
% PVGUI MATLAB code for pvGUI.fig
%      PVGUI, by itself, creates a new PVGUI or raises the existing
%      singleton*.
%
%      H = PVGUI returns the handle to a new PVGUI or the handle to
%      the existing singleton*.
%
%      PVGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PVGUI.M with the given input arguments.
%
%      PVGUI('Property','Value',...) creates a new PVGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pvGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pvGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pvGUI

% Last Modified by GUIDE v2.5 07-Jul-2022 10:45:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @pvGUI_OpeningFcn, ...
  'gui_OutputFcn',  @pvGUI_OutputFcn, ...
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
function pvGUI_OpeningFcn(hObject, ~, handles, varargin)
handles.output = hObject;

handles.hGUI = handles.figure1;
handles.the_project = -1;
handles.the_experiment = -1;

handles.root = fileparts(which('pvGUI'));

local_root = [handles.root, filesep, 'Local'];
if exist(handles.root, 'file') == 7
  handles.root = local_root;
  % load ini file
  ini_file = fullfile(handles.root, 'pvGUI.ini');
  if exist(ini_file, 'file') == 2
    ini = inimanage(ini_file);
    handles.root = safeget(ini.directories, 'root', []);
    handles.drive = safeget(ini.directories, 'drive', []);
  else
    ini.directories.root  = handles.root;
    ini.directories.drive = handles.root(1);
    inimanage(ini_file, ini);
  end
else
    handles.drive = handles.root(1);
end

% load project list
oldpath = my_cd(handles.root);
fname_project_list = fullfile(handles.root, 'project_list.m');
if exist(fname_project_list, 'file') ~= 2
  fid = fopen(fname_project_list, 'a+');
  if fid ~= -1
    fprintf(fid, 'function projects = project_list\n\n');
    fprintf(fid, 'projects = {};\n');
    fprintf(fid, 'projects{end+1}.title=''first'';\n');
    fprintf(fid, 'projects{end}.folder=''first'';\n');
    fclose(fid); 
  end
end
% handles.projects = eval(project_list);
handles.projects = project_list;

cd(oldpath);

str = {};
for ii=1:length(handles.projects)
  if ~isempty(strfind(handles.projects{ii}.title, '----'))
    str{end+1} = ['<html><font color="blue"><b>',handles.projects{ii}.title, '<b></font></html>'];
  else
    str{end+1} = handles.projects{ii}.title;
  end
end
set(handles.pmProject, 'string', str);

handles.ActiveReports = [];

% Update handles structure
guidata(hObject, handles);

pmProject_Callback(hObject, [], handles);

% --------------------------------------------------------------------
function old_path = my_cd(new_path)
old_path = pwd;
cd(new_path);

% --------------------------------------------------------------------
function varargout = pvGUI_OutputFcn(~, ~, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pmProject_Callback(hObject, ~, handles)
handles.the_project = get(handles.pmProject, 'value');
handles.p_path = fullfile(handles.root, handles.projects{handles.the_project}.folder);
if strfind(handles.projects{handles.the_project}.folder, '------'), return; end
handles.the_experiment = -1;

current_directory = pwd;
if exist(handles.p_path, 'file') ~= 7 
  fprintf('Directory %s not found.\n', handles.p_path);
  
  ButtonName = questdlg(sprintf('Directory %s not found. Create new project ?', handles.p_path), ...
    'New project ?', ...
    'Yes', 'No', 'Cancel');
  switch ButtonName
    case 'Yes'
      % create directory
      cd(handles.root); mkdir(handles.projects{handles.the_project}.folder); cd(current_directory);
    case 'No', return;
    case 'Cancel', return;
  end
end
cd(handles.p_path);

if exist(fullfile(handles.p_path, 'pv_database.m'), 'file') ~= 2 
  ButtonName = questdlg('Database file is not found. Create new database ?', ...
    'New project ?', ...
    'Yes', 'No', 'Cancel');
  switch ButtonName
    case 'Yes'
      % create files
      fname = fullfile(handles.p_path, 'pv_database.m');
      fid = fopen(fname, 'a+');
      if fid ~= -1, fprintf(fid, 'function expN = pv_database()\n\nexpN={};\n'); fclose(fid); end
    case 'No', return;
    case 'Cancel', return;
  end
end
handles.experiments = pv_database;

if exist(fullfile(handles.p_path, 'pv_draw_item.m'), 'file') ~= 2
  % create files
  fname = fullfile(handles.p_path, 'pv_draw_item.m');
  fid = fopen(fname, 'a+');
  if fid ~= -1, fprintf(fid, 'function item_text = pv_draw_item(the_item)\n\nitem_text = the_item.tag;\n'); fclose(fid); end
end

if exist(fullfile(handles.p_path, 'pv_item_comment.m'), 'file') ~= 2
  % create files
  fname = fullfile(handles.p_path, 'pv_item_comment.m');
  fid = fopen(fname, 'a+');
  if fid ~= -1, fprintf(fid, 'function res = pv_item_comment(the_item)\n\nres = [''registration:'', the_item.registration];\n'); fclose(fid); end
end

if exist(fullfile(handles.p_path, 'pv_analysis.m'), 'file') ~= 2
  % create files
  fname = fullfile(handles.p_path, 'pv_analysis.m');
  fid = fopen(fname, 'a+');
  if fid ~= -1, fprintf(fid, 'function res = pv_analysis\n\n'); fclose(fid); end
end

str = {};
for ii=1:length(handles.experiments)
  str{end+1} = pv_draw_item(handles.experiments{ii});
end
set(handles.lbExp, 'string', str, 'value', 1);

guidata(hObject, handles);
cd(current_directory);

% --------------------------------------------------------------------
function cbLoadRegistration_Callback(~, ~, handles)

if ~isfield(handles.experiments{handles.the_experiment}, 'registration') || ...
    isempty(handles.experiments{handles.the_experiment}.registration)
  arbuz_ShowMessage(handles.hGUI, sprintf('Experiment %s has no registration.', handles.experiments{handles.the_experiment}.tag));
  set(handles.lbImages, 'string', {'No data'}, 'value', 1);
  cd(current_directory);
  return;
end

arbuz_OpenProject(handles.figure1, handles.experiments{handles.the_experiment}.registration);

res = arbuz_FindImage(handles.figure1, 'master', '', '', {'Name'});
str = {};
for ii=1:length(res)
  str{end+1} = res{ii}.Name;
end
set(handles.lbImages, 'string',str);
tag = safeget(handles.experiments{handles.the_experiment}, 'tag', '');
set(handles.uipanel1, 'title',['Registration browser - ',tag,]);


saves = arbuz_GetSaveParList(handles.figure1, '');
if isempty(saves)
  set(handles.pmReportView, 'string', {'No scenes found.'}, 'value', 1)
else
  set(handles.pmReportView, 'string', saves, 'value', 1)
end

% --------------------------------------------------------------------
function lbExp_Callback(hObject, ~, handles)
handles.the_experiment = get(handles.lbExp, 'value');
guidata(hObject, handles);

current_directory = pwd;
cd(handles.p_path);

% Load project specific comments
res = pv_item_comment(handles.experiments{handles.the_experiment});
set(handles.eComments, 'string', res);

% verify registration
registration = safeget(handles.experiments{handles.the_experiment}, 'registration', '');
if iscell(registration)
else
    registration = {registration};
end
registration = change_drive(registration, handles.drive);
set(handles.lbRegistrations, 'String', registration, 'Value', 1);

rrg = registration{1};
if exist(rrg, 'file') == 2 || exist([rrg,'.mat'], 'file') == 2
    handles.pbArbuzGUI.FontWeight = 'bold';
    handles.pbArbuzGUI.Enable = 'on';
else
    handles.pbArbuzGUI.FontWeight = 'normal';
    handles.pbArbuzGUI.Enable = 'off';
end

cd(current_directory);
% --------------------------------------------------------------------

function out = change_drive(in, new_drive)
out=in;

if ~isempty(new_drive) && length(new_drive) == 1
  for ii=1:length(out)
    pos = strfind(out{ii}, ':\');
    if ~isempty(pos) && pos == 2
      ppp = out{ii}; ppp(1) = new_drive; out{ii} = ppp;
    end
  end
end
% --------------------------------------------------------------------
function lbImages_Callback(~, ~, handles)

% --------------------------------------------------------------------
function pbViewer_Callback(~, ~, handles)

the_image = get(handles.lbImages, 'value');
the_image_str = get(handles.lbImages, 'string');
res = arbuz_FindImage(handles.figure1, 'master', 'Name', the_image_str{the_image}, {'Name'});
options.image.ImageIdx = res{1}.ImageIdx;
options.image.SlaveIdx = -1;
arbuz_DefaultRDR(handles.hGUI, options);

% --------------------------------------------------------------------
function pbLoadOriginal_Callback(hObject, ~, handles)
the_image = get(handles.lbImages, 'value');
the_image_str = get(handles.lbImages, 'string');
res = arbuz_FindImage(handles.figure1, 'master', 'Name', the_image_str{the_image}, {'Name', 'FileName'});
ibGUI(res{1}.FileName)

% --------------------------------------------------------------------
function pbArbuzGUI_Callback(hObject, ~, handles)
if handles.the_experiment < 0, return; end
registrations = handles.experiments{handles.the_experiment}.registration;
registrations = change_drive(registrations, handles.drive);
if iscell(registrations)
  registration = registrations{handles.lbRegistrations.Value};
else
  registration = {registrations};
end
ArbuzGUI([], registration);

% --------------------------------------------------------------------
function pb3D_Callback(hObject, ~, handles)
arbuz_RedrawAll(handles.figure1)

% --------------------------------------------------------------------
function pbReportView_Callback(hObject, ~, handles)
val = get(handles.pmReportView, 'value');
str = get(handles.pmReportView, 'string');

the_save = arbuz_GetSavePar(handles.figure1, str{val});
if ~isempty(the_save)
  eval([the_save.renderer, '(handles.figure1, the_save)'])
end

% --------------------------------------------------------------------
function eComments_Callback(hObject, ~, handles)
% hObject    handle to eComments (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of eComments as text
%        str2double(get(hObject,'String')) returns contents of eComments as a double

% --------------------------------------------------------------------
function pbProcessAllData_Callback(hObject, ~, handles)
handles.p_path = fullfile(handles.root, handles.projects{handles.the_project}.folder);
cd(handles.p_path);

try
  all_processing = pv_analysis;
catch
  disp('Processing is not defined for this project.');
  return;
end

str = {};
for ii=1:length(all_processing), str{end+1} = all_processing{ii}.title; end 


%Project_num = handles.the_project;


[sel,isOK] = listdlg('PromptString','Select processing:',...
  'SelectionMode','single','ListSize', [180,250], ...
  'ListString',str);

isReprocess = true;

if isOK
  parameters = all_processing{sel}.parameters;
  parameters.path = handles.p_path;
  parameters.reprocess = isReprocess;
  
  switch(hObject)
    case handles.pbProcessSingle
      parameters.experiment = 1;
      eval([all_processing{sel}.function,'(handles.experiments(handles.the_experiment),parameters);']);
    otherwise
      parameters.experiment = handles.the_experiment;
      eval([all_processing{sel}.function,'(handles.experiments,parameters);']);
  end
end

% --------------------------------------------------------------------
function pbProcessSingle_Callback(hObject, ~, handles)
% hObject    handle to pbProcessSingle (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --------------------------------------------------------------------
function pbDBGUI_Callback(hObject, ~, handles)
pvDBGUI(handles);

% --------------------------------------------------------------------
function pmShowGroup_Callback(hObject, ~, handles)
% hObject    handle to pmShowGroup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns pmShowGroup contents as cell array
%        contents{get(hObject,'Value')} returns selected item from pmShowGroup


% --------------------------------------------------------------------
function pmReportView_Callback(~, ~, handles)


% --------------------------------------------------------------------
function mEdit_Callback(hObject, ~, handles)

% external editor
editor = System.Diagnostics.Process();
editor.StartInfo.FileName = 'notepad.exe';
if exist('c:\Program Files (x86)\Notepad++\notepad++.exe', 'file'), editor.StartInfo.FileName = '"c:\Program Files (x86)\Notepad++\notepad++.exe"'; end

switch hObject
  case handles.mEditAnalysis
    the_sel = get(handles.pmProject, 'value');
    the_path = fullfile(handles.root, handles.projects{the_sel}.folder);
    if exist(the_path, 'file') ~= 7, return; end
    argument = fullfile(the_path,'pv_analysis.m');
    editor.StartInfo.Arguments = argument;
    %     Start(editor);
    backup_file(argument)
    eval(['edit ' , argument]);
  case handles.mEditDrawScript
    the_sel = get(handles.pmProject, 'value');
    the_path = fullfile(handles.root, handles.projects{the_sel}.folder);
    if exist(the_path, 'file') ~= 7, return; end
    argument = fullfile(the_path,'pv_draw_item.m');
    editor.StartInfo.Arguments = argument;
    %     Start(editor);
    backup_file(argument)
    eval(['edit ' , argument]);
  case handles.mEditCommentScript
    the_sel = get(handles.pmProject, 'value');
    the_path = fullfile(handles.root, handles.projects{the_sel}.folder);
    if exist(the_path, 'file') ~= 7, return; end
    argument = fullfile(the_path,'pv_item_comment.m');
    editor.StartInfo.Arguments = argument;
%     Start(editor);
    backup_file(argument)
    eval(['edit ' , argument]);
  case handles.mEditExerimentList
    the_sel = get(handles.pmProject, 'value');
    the_path = fullfile(handles.root, handles.projects{the_sel}.folder);
    if exist(the_path, 'file') ~= 7
      error('File does not exist.');
    else
      argument = fullfile(the_path,'pv_database.m');
      editor.StartInfo.Arguments = argument;
      %     Start(editor);
      backup_file(argument)
      eval(['edit ' , argument]);
    end
  case handles.mEditProjectList
    argument = fullfile(handles.root,'project_list.m');
    editor.StartInfo.Arguments = argument;
%     Start(editor);
    backup_file(argument)
    eval(['edit ' , argument]);
end

% --------------------------------------------------------------------
function backup_file(file_name)
pvGUI_filename = which(mfilename);
[pvGUI_folder] = fileparts(pvGUI_filename);

backup_folder = fullfile(pvGUI_folder, 'backup');
if ~exist(backup_folder, 'dir')
  mkdir(backup_folder);
end

if contains(file_name, pvGUI_folder)
  pos = strfind(file_name, pvGUI_folder);
  backup_file = file_name(pos+length(pvGUI_folder):end);
  backup_file(backup_file == '\') = '_';
  backup_file = fullfile(backup_folder, [datestr(now, 'HHMMSS-yymmdd'),backup_file]);
  copyfile(file_name, backup_file);
end

% --------------------------------------------------------------------
function pbOpenFolder_Callback(hObject, ~, handles)
registration = safeget(handles.experiments{handles.the_experiment}, 'registration', '');
if iscell(registration)
else
    registration = {registration};
end

rrg = registration{handles.lbRegistrations.Value};
if exist(rrg, 'file') == 2 || exist([rrg,'.mat'], 'file') == 2
    handles.pbArbuzGUI.FontWeight = 'bold';
    handles.pbArbuzGUI.Enable = 'on';
else
    handles.pbArbuzGUI.FontWeight = 'normal';
    handles.pbArbuzGUI.Enable = 'off';
end

% --- Executes on button press in cbReprocess.
function cbReprocess_Callback(hObject, ~, handles)
% hObject    handle to cbReprocess (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of cbReprocess
