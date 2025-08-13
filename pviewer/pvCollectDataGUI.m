function varargout = pvCollectDataGUI(varargin)
% PVCOLLECTDATAGUI MATLAB code for pvCollectDataGUI.fig
%      PVCOLLECTDATAGUI, by itself, creates a new PVCOLLECTDATAGUI or raises the existing
%      singleton*.
%
%      H = PVCOLLECTDATAGUI returns the handle to a new PVCOLLECTDATAGUI or the handle to
%      the existing singleton*.
%
%      PVCOLLECTDATAGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PVCOLLECTDATAGUI.M with the given input arguments.
%
%      PVCOLLECTDATAGUI('Property','Value',...) creates a new PVCOLLECTDATAGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pvCollectDataGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pvCollectDataGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pvCollectDataGUI

% Last Modified by GUIDE v2.5 25-Oct-2019 10:41:04

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pvCollectDataGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @pvCollectDataGUI_OutputFcn, ...
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


% --- Executes just before pvCollectDataGUI is made visible.
function pvCollectDataGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pvCollectDataGUI (see VARARGIN)

% Choose default command line output for pvCollectDataGUI
handles.output = hObject;

if nargin > 4
  handles.the_list = varargin{2};
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pvCollectDataGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = pvCollectDataGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
the_list = handles.the_list;

ouput_data = {};
for ii=1:length(the_list)
  if isfield(the_list{ii}, 'ExpError')
    is_error = safeget(the_list{ii}, 'ExpError', 5) > 0;
  elseif isfield(the_list{ii}, 'Censor_Code')
     code = safeget(the_list{ii}, 'Censor_Code', 5);
     is_error = code == 5 || code==6;
  else
  end
  if(is_error > 0), continue; end
  
  registration = safeget(the_list{ii}, 'registration', '');
  if(isempty(registration)), continue; end
  tag = safeget(the_list{ii}, 'tag', '');
  
  % 1. loading registration
  fprintf('%d Loading data:%s\n', ii, registration);
  
  arbuz_OpenProject(handles.figure1, registration);
  arbuz_ApplyTransformation(handles.figure1, '', 'fix');
  
  fprintf('  -- Extracting pO2\n', ii);
  try
    ipO2 = arbuz_FindImage(handles.figure1, 'master', 'InName', 'PO2', {'data','Ashow','Mask'});
  catch
    fprintf('  -- ERROR pO2\n', ii);
  end
  
  res = [];
  for jj=1:length(ipO2)
    res{end+1}.pO2 = ipO2{jj}.data;
    res{end}.Name = ipO2{jj}.Image;
    res{end}.Mask = ipO2{jj}.Mask;
  end
  
  fprintf('  -- Storing data\n', ii);
  ouput_data{end+1}.images = res;
  
  fprintf('  -- Loading mask\n', ii);
  try
    im_list = arbuz_FindImage(handles.figure1, 'master', 'ImageType', 'MRI', {'FileName','slavelist'});
    im_MRI = arbuz_FindImage(handles.figure1, im_list, 'InName', 'ax', {'FileName','slavelist'});
    image_slaves = arbuz_FindImage(handles.figure1, im_MRI{1}.SlaveList, 'ImageType', '3DMASK', {'data', 'AShow'});
    image_MRI_tumor = arbuz_FindImage(handles.figure1, image_slaves, 'Name', 'tumor', {'data', 'AShow'});
    if isempty(image_MRI_tumor)
      image_MRI_tumor = arbuz_FindImage(handles.figure1, image_slaves, 'InName', 'umor', {'data', 'AShow'});
    end
    MRI_tumor_atEPR = arbuz_util_transform(handles.figure1, image_MRI_tumor{1}, ipO2{1}, []);
    fprintf('  -- Storing mask\n', ii);
    ouput_data{end}.TumorMask = MRI_tumor_atEPR.data;
  catch
    fprintf('  -- ERROR Tumor Mask\n', ii);
    ouput_data{end}.TumorMask = [];
  end
  
end

file_name = fullfile(get(handles.eOutputFolder,'String'),['data_',tag,'.mat']);
fprintf('Saving file %s\n', file_name);
o1.data = ouput_data;
save(file_name, '-struct', 'o1');

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
[PathName] = uigetdir();
if ~isequal(PathName, 0)
  set(handles.eOutputFolder, 'String', PathName);
end
