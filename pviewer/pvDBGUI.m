function varargout = pvDBGUI(varargin)
% PVDBGUI MATLAB code for pvDBGUI.fig
%      PVDBGUI, by itself, creates a new PVDBGUI or raises the existing
%      singleton*.
%
%      H = PVDBGUI returns the handle to a new PVDBGUI or the handle to
%      the existing singleton*.
%
%      PVDBGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PVDBGUI.M with the given input arguments.
%
%      PVDBGUI('Property','Value',...) creates a new PVDBGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pvDBGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pvDBGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pvDBGUI

% Last Modified by GUIDE v2.5 04-Oct-2016 14:25:08

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @pvDBGUI_OpeningFcn, ...
  'gui_OutputFcn',  @pvDBGUI_OutputFcn, ...
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


% --- Executes just before pvDBGUI is made visible.
function pvDBGUI_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;

handles.hGUI = handles.figure1;
handles.the_project = -1;
handles.the_experiment = -1;
handles.access = 0;
handles.experiments = {};
handles.fields = {};

if length(varargin)>0
handles.pmProject.Value = varargin{1}.pmProject.Value;
end


% load project list
project_list;
handles.projects = EPRproject;
str = {};
for ii=1:length(handles.projects)
  str{end+1} = handles.projects{ii}.title;
end
set(handles.pmProject, 'string', str);

% find the root
handles.root = fileparts(get(handles.figure1, 'FileName'));

handles.ActiveReports = [];


% Update handles structure
guidata(hObject, handles);
set_access(handles, handles.access);
pmProject_Callback(hObject, [], handles);



% --- Outputs from this function are returned to the command line.
function varargout = pvDBGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function lbValues_Callback(hObject, eventdata, handles)
exp = get(handles.lbExp, 'value');
fld = get(handles.lbValues, 'value');
set(handles.eValue, 'string', get_field_as_string(handles, exp, fld))
access = strcmp(handles.fields{fld}.edit, 'yes');
set_access_field(handles, access || handles.access);

% --------------------------------------------------------------------
function set_fields(handles)
exp = get(handles.lbExp, 'value');
fld = get(handles.lbValues, 'value');
fld = max(fld, 1);
% list = cellfun( @(x) x.name, handles.fields, 'UniformOutput', false );

list = {};
for ii=1:length(handles.fields) 
  decor = '</html>'; decol = '<html><b>'; decom = '</b>';
  if strcmp(handles.fields{ii}.edit, 'yes')
    decol = '<html><font color=blue><b>';
    decom = '</b></font>';
    decor = '</html>';
  end
  list{end+1} = [decol, handles.fields{ii}.name,decom,'   -   ',get_field_as_string(handles, exp, ii), decor];
end
fld = min(fld, length(list));
set(handles.lbValues, 'string', list, 'value', fld);

% --------------------------------------------------------------------
function pbSave_Callback(hObject, eventdata, handles)
Save(handles, 'C:/Temp/dbase.m');

% --------------------------------------------------------------------
function pmProject_Callback(hObject, eventdata, handles)
handles.the_project = get(handles.pmProject, 'value');
handles.p_path = fullfile(handles.root, handles.projects{handles.the_project}.folder);
handles.the_experiment = -1;

current_directory = pwd;
cd(handles.p_path);

try
  [handles.experiments, handles.fields] = pv_database;
catch
  handles.experiments = pv_database;
  handles.fields = {};
  disp('No field description found.');
end
set_fields(handles);
set_experiments(handles);

guidata(hObject, handles);
cd(current_directory);

function set_experiments(handles)
str = {};
for ii=1:length(handles.experiments)
  str{end+1} = pv_draw_item(handles.experiments{ii});
end
set(handles.lbExp, 'string', str, 'value', 1);

% --------------------------------------------------------------------
function lbExp_Callback(hObject, eventdata, handles)
handles.the_experiment = get(handles.lbExp, 'value');
guidata(hObject, handles);

current_directory = pwd;
cd(handles.p_path);

% Load project specific comments
res = pv_item_comment(handles.experiments{handles.the_experiment});
set(handles.eComments, 'string', res);

cd(current_directory);

set_fields(handles);
lbValues_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function eValue_Callback(hObject, eventdata, handles)
exp = get(handles.lbExp, 'value');
fld = get(handles.lbValues, 'value');
str = strtrim(get(handles.eValue, 'string'));

switch handles.fields{fld}.type
  case 'S', handles.experiments{exp}.(handles.fields{fld}.name) = str;
  otherwise
    handles.experiments{exp}.(handles.fields{fld}.name) = str2num(str);
end
guidata(hObject, handles);
set_fields(handles);

% --------------------------------------------------------------------
function pbNewField_Callback(hObject, eventdata, handles)
% request the field
res = inputdlg({'Enter the name of the new field', 'String (S) or Number (N)', 'Editable yes / no', }, ...
  'Name', 1, {'', 'N', 'yes'});
if ~isempty(res)
  
  name = strtrim(res{1});
  % check for the field
  for jj=1:length(handles.fields)
    if strcmpi(handles.fields{jj}.name, name)
      disp('Field name alreay exists.');
      return;
    end
  end
    
    % add the field
    handles.fields{end+1} = struct('name', name, 'type', res{2}, 'edit', res{3}); 
  
  
  guidata(handles.figure1, handles);
  set_fields(handles);
  lbValues_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function pbDeleteField_Callback(hObject, eventdata, handles)
fld = get(handles.lbValues, 'value');
access = strcmp(handles.fields{fld}.edit, 'yes');
if(~(access || handles.access)), return; end

idx = 1:length(handles.fields);
handles.fields = handles.fields(idx~=fld);
guidata(handles.figure1, handles);

set_fields(handles);
lbValues_Callback(hObject, eventdata, handles);

% --------------------------------------------------------------------
function Save(handles, filename)

f = fopen(filename, 'w+');

fprintf(f, '% pvGUI database\n');
fprintf(f, 'function [expN, dsc] = pv_database()\n\n');
fprintf(f, 'expN={};\n\n');

for ii=1:length(handles.experiments)
  fprintf(f, '%% "%s" -------------------------\n', safeget(handles.experiments{ii}, 'tag', ''));
  for jj=1:length(handles.fields)
    fprintf(f, 'expN{end%s}.%s = %s;\n', iff(jj==1, '+1', ''), ...
      handles.fields{jj}.name, ...
      get_field_4_saving(handles, ii, jj));
  end
  fprintf(f, '\n');
end

fprintf(f, '\ndsc = {};\n');
fprintf(f, 'df = @(name, thetype, isedit) struct(''name'', name, ''type'', thetype, ''edit'', isedit);\n\n');

for jj=1:length(handles.fields)
  fprintf(f, 'dsc{end%s} = df(''%s'',''%s'',''%s'');\n', iff(jj==1, '+1', ''), ...
    handles.fields{jj}.name, ...
    handles.fields{jj}.type, ...
    handles.fields{jj}.edit);
end

fclose(f);

% --------------------------------------------------------------------
function string_value = get_field_as_string(handles, ii, jj)
value = safeget(handles.experiments{ii}, handles.fields{jj}.name, '');

if ischar(value)
  string_value = value;
else
  string_value = ['[',num2str(value),']'];
  string_value = strrep(string_value, '    ',' ');
  string_value = strrep(string_value, '   ',' ');
  string_value = strrep(string_value, '  ',' ');
end

% --------------------------------------------------------------------
function string_value = get_field_4_saving(handles, ii, jj)
string_value = get_field_as_string(handles, ii, jj);

if handles.fields{jj}.type == 'S'
  string_value = ['''',string_value,''''];
elseif isempty(string_value)
  string_value = ['[',string_value,']'];
end

% --------------------------------------------------------------------
function pbChangeField_Callback(hObject, eventdata, handles)
fld = get(handles.lbValues, 'value');
% request the field
fields = {}; for ii=1:length(handles.fields), fields{end+1} = handles.fields{ii}.name; end
expression = pvExpressionEditor([],fields);

for ii=1:length(handles.experiments)
  newexpression = expression;
  for jj=1:length(handles.fields)
    if strfind(expression, handles.fields{jj}.name)
      if handles.fields{jj}.type == 'N'
        newexpression = strrep(newexpression, handles.fields{jj}.name, ...
          num2str(handles.experiments{ii}.(handles.fields{jj}.name)));
      elseif handles.fields{jj}.type == 'S'
        newexpression = strrep(newexpression, handles.fields{jj}.name, ...
          ['''',handles.experiments{ii}.(handles.fields{jj}.name),'''']);
      end
    end
  end
  
  handles.experiments{ii}.(handles.fields{fld}.name) = ...
    eval(newexpression);
end

guidata(hObject, handles);
lbValues_Callback(hObject, eventdata, handles);


% check the fields
function res = is_dbfield(handles, fld)
res = false;
for ii=1:length(handles.fields)
  if strcmp(handles.fields{ii}.name,fld) == true
    res = true;
    return;
  end
end

% --- Executes on button press in pbDataAnalyse.
function pbDataAnalyse_Callback(hObject, eventdata, handles)
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
  'SelectionMode','single','ListSize', [160,120], ...
  'ListString',str);

if isOK
  parameters = all_processing{sel}.parameters;
  
  parameters.experiment = handles.the_experiment;
  eval([all_processing{sel}.function,'(handles.experiments,parameters);']);
end
% --------------------------------------------------------------------

function set_access(handles, access)
state = iff(access, 'on', 'off');
set([handles.mRecordAdd, handles.mRecordDelete], 'Enable', state)
%lbValues_Callback([], [], handles);
% --------------------------------------------------------------------

function set_access_field(handles, access)
state = iff(access, 'on', 'off');
set([handles.eValue, handles.mFieldEvaluate, handles.pbChangeField,...
  handles.mFieldDelete, handles.pbDeleteField], 'Enable', state)

% --------------------------------------------------------------------
function mFileSave_Callback(hObject, eventdata, handles)
% hObject    handle to mFileSave (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function mFieldAdd_Callback(hObject, eventdata, handles)
% hObject    handle to mFieldAdd (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function mFieldDelete_Callback(hObject, eventdata, handles)
% hObject    handle to mFieldDelete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function mFieldEvaluate_Callback(hObject, eventdata, handles)
% hObject    handle to mFieldEvaluate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function mAccessLogin_Callback(hObject, eventdata, handles)
handles.access = 0;
pass = passcode;

if ~isempty(pass) && strcmp(pass, 'itsme')
    handles.access = 1;
end
guidata(hObject, handles);
set_access(handles, handles.access);


% --------------------------------------------------------------------
function mRecordAdd_Callback(hObject, eventdata, handles)

res = inputdlg('Tag', 'New record');

if ~isempty(res)
  exp.tag = res{1};
  handles.experiments{end+1} = exp;
  guidata(hObject, handles);
  set_experiments(handles);
end

% --------------------------------------------------------------------
function mRecordDelete_Callback(hObject, eventdata, handles)
% hObject    handle to mRecordDelete (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes when figure1 is resized.
function figure1_SizeChangedFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
