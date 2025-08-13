function varargout = INNARPT(varargin)
% INNARPT MATLAB code for INNARPT.fig
%      INNARPT, by itself, creates a new INNARPT or raises the existing
%      singleton*.
%
%      H = INNARPT returns the handle to a new INNARPT or the handle to
%      the existing singleton*.
%
%      INNARPT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in INNARPT.M with the given input arguments.
%
%      INNARPT('Property','Value',...) creates a new INNARPT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before INNARPT_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to INNARPT_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help INNARPT

% Last Modified by GUIDE v2.5 12-Nov-2018 14:53:48

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @INNARPT_OpeningFcn, ...
                   'gui_OutputFcn',  @INNARPT_OutputFcn, ...
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
function INNARPT_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to INNARPT (see VARARGIN)

% Choose default command line output for INNARPT
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};

% Fill Reference
fill_PopupMenu(handles, handles.pmReference, 'master', false);

% Fill PET
fill_PopupMenu(handles, handles.pmPET, 'master', false);

handles.set{1} = struct('name', 'Col1', 'min', 0, 'max',20, 'value', 0);
handles.set{2} = struct('name', 'Col2', 'min', 0, 'max',20, 'value', 0);
handles.set{3} = struct('name', 'Col3', 'min', 0, 'max',20, 'value', 0);
handles.set{4} = struct('name', 'Col4', 'min', 0, 'max',20, 'value', 6);
handles.set{5} = struct('name', 'Col5', 'min', 0, 'max',20, 'value', 6);
handles.set{6} = struct('name', 'Col6', 'min', 0, 'max',20, 'value', 6);
handles.set{7} = struct('name', 'Col7', 'min', -90, 'max',90, 'value', 0);
handles.set{8} = struct('name', 'Col8', 'min', -90, 'max',90, 'value', 0);

row_shift = 2.2;
pos = handles.pControls.Position;
row_top = pos(4) - 2;

for ii=1:length(handles.set)
  row  = row_top - ii *row_shift;
  handles.set{ii}.tcontrols = uicontrol('Style','text', 'Parent', handles.pControls,  ...
    'Units', 'characters', 'Position', [1, row, 5, 2], 'string', handles.set{ii}.name, ...
    'Callback', '');
  handles.set{ii}.pbcontrol = uicontrol('Style','popupmenu', 'Parent', handles.pControls,  ...
    'Units', 'characters', 'Position', [9, row+0.3, 25, 2], 'string', 'empty');
  fill_PopupMenu(handles, handles.set{ii}.pbcontrol, 'master', true);
end

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes INNARPT wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = INNARPT_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pmReference_Callback(hObject, eventdata, handles)
items = handles.pmReference.String;
item = items{handles.pmReference.Value};

fill_PopupMenu(handles, handles.pmMask, item, true)

% --------------------------------------------------------------------
function pbProcess_Callback(hObject, eventdata, handles)
set(handles.eLog, 'string', 'Processing...', 'backgroundcolor', [1 0 0]);
drawnow;

data_output = {};
for ii=1:length(handles.set)
  data_output{ii}.Normalization = 1;
  data_output{ii}.data = [];
end

% data_location = {};

% reference frame
item = handles.pmReference.String{handles.pmReference.Value};
Reference = arbuz_FindImage(handles.hh, 'master', 'Name', item, {'Name','SlaveList','data','bbox','Ashow'});
item = handles.pmMask.String{handles.pmMask.Value};
TumorMaskID = arbuz_FindImage(handles.hh, Reference{1}.SlaveList, 'Name', item, {'Name','data'});
TumorMask = TumorMaskID{1}.data > 0.5;

se = epr_strel('cube', 1);
ExtendedTumorMask = imdilate(TumorMask, se);

% PET normalization
pet_item = handles.pmPET.String{handles.pmPET.Value};
PET = arbuz_FindImage(handles.hh, 'master', 'Name', pet_item, {'Name','SlaveList','data','bbox','Ashow'});
item = handles.pmPETMask.String{handles.pmPETMask.Value};
PETMuscle = arbuz_FindImage(handles.hh, PET{1}.SlaveList, 'Name', item, {'Name','data'});
handles.MeanPETMuscle = 1;
if ~isempty(PET) && ~isempty(PETMuscle)
  handles.MeanPETMuscle = mean(PET{1}.data(PETMuscle{1}.data));
  fprintf('PET average value in the mask is %g\n', handles.MeanPETMuscle);
  
  for ii=1:length(handles.set)
    item = handles.set{ii}.pbcontrol.String{handles.set{ii}.pbcontrol.Value};
    if contains(item, pet_item)
      data_output{ii}.Normalization = handles.MeanPETMuscle;
    end
  end
end

% other images
report_string = '';
if ~isempty(Reference)
  % eliminate data where oxygen image was not obtained
  if contains(Reference{1}.ImageType, 'PO2_pEPRI')
    fprintf('Words of wisdom:\n');
    fprintf('Tumor: %i vox\n', numel(find(TumorMask)));
    fprintf('Extended: %i vox\n', numel(find(ExtendedTumorMask)));

%     TumorMask = TumorMask & Reference{1}.data > -25;
%     ExtendedTumorMask = ExtendedTumorMask & Reference{1}.data > -25;
    
    fprintf('Tumor restricted: %i vox\n', numel(find(TumorMask)));
    fprintf('Extended restricted: %i vox\n', numel(find(ExtendedTumorMask)));
  end
  
  for ii=1:length(handles.set)
      item = handles.set{ii}.pbcontrol.String{handles.set{ii}.pbcontrol.Value};
      if isempty(strfind(item, 'not use'))
          if strfind(item,'pO2')==1
              Source = arbuz_FindImage(handles.hh, 'master', 'Name', item, {'data'});
%               res = arbuz_util_transform(handles.hh, Source{1}, Reference{1}, []);
              GoodData = Source{1}.data > -25;
              res = arbuz_util_transform(handles.hh, Source{1}, Reference{1}, GoodData);

              TumorMask = TumorMask & res.data > 0.5;
              ExtendedTumorMask = ExtendedTumorMask & res.data > 0.5;
          end
      end
  end
  for ii=1:length(handles.set)
    item = handles.set{ii}.pbcontrol.String{handles.set{ii}.pbcontrol.Value};
    if isempty(strfind(item, 'not use'))
      Source = arbuz_FindImage(handles.hh, 'master', 'Name', item, {});
      res = arbuz_util_transform(handles.hh, Source{1}, Reference{1}, []);
%       if strfind(item,'pO2')==1
%           TumorMask = TumorMask & res.data > -25;
%           ExtendedTumorMask = ExtendedTumorMask & res.data > -25;
%       end
      sz4 = size(res.data, 4);
      for jj=1:sz4
        input_data = res.data(:,:,:,jj);
        data_output{ii}.data(:,jj) = input_data(TumorMask);
        data_output{ii}.extdata(:,jj) = input_data(ExtendedTumorMask);
      end
      report_string = sprintf('%s %s (%i) ', report_string, item, size(data_output{ii}.data, 1));
    end
  end
end 

handles.data_output = data_output;
handles.data_mask   = TumorMask;
global_index = find(TumorMask);
[A,B,C]  = ind2sub(size(TumorMask), global_index);
handles.data_pos = [A,B,C];
global_index = find(ExtendedTumorMask);
[A,B,C]  = ind2sub(size(ExtendedTumorMask), global_index);
handles.ext_data_pos = [A,B,C];
% markers = double(TumorMask);
% markers(ExtendedTumorMask & ~TumorMask) = 2;
% handles.markers(:,jj) = markers(ExtendedTumorMask);

guidata(handles.figure1, handles);


fprintf(1, 'Arrays: %s\n', report_string);
disp('We are done! Display results to see results.');
set(handles.eLog, 'string', 'Finished!', 'backgroundcolor', [0 1 0]);

assignin('base', 'results', data_output); %, data_location);
% evalin('base', 'disp(results)');

% --------------------------------------------------------------------
function pbExport_Callback(hObject, eventdata, handles)
data_output = handles.data_output;

ndata = 1;
ncol  = 0;

dataindex = false(length(data_output), 1);
switch 1
  case 1 % use only tuimor voxels
    % calculate number of columns
    for ii=1:length(data_output)
      if ~isempty(data_output{ii}.data)
        dataindex(ii) = true;
        ncol = ncol + 1;
        ndata = length(data_output{ii}.data);
      end
    end
    
    output = zeros(ndata, ncol+3);
    
    output(:,1:3) = handles.data_pos;
    ncol = 1;
    for ii=1:length(dataindex)
      if dataindex(ii)
        output(:,ncol+3) = data_output{ii}.data(:,1);
        ncol = ncol + 1;
      end
    end
  case 2 % add surrounding voxels
    % calculate number of columns
    for ii=1:length(data_output)
      if ~isempty(data_output{ii}.data)
        dataindex(ii) = true;
        ncol = ncol + 1;
      end
    end
    ndata = length(handles.markers);
    
    output = zeros(ndata, ncol+4);
    
    output(:,1:3) = handles.ext_data_pos;
    output(:,4) = handles.markers;
    ncol = 1;
    for ii=1:length(dataindex)
      if dataindex(ii)
        output(:,ncol+4) = data_output{ii}.extdata(:,1);
        ncol = ncol + 1;
      end
    end    
end

% data_location = handles.data_location;
assignin('base', 'results', output); %, data_location);
disp('Data are stored in results.');


fid = fopen('X:\Inna\projects\FMISO\tmp.csv', 'w+');

switch 1
  case 1
    for jj=1:size(handles.data_pos, 1)
      data_pos = (handles.data_pos * 0.6629)-((0.6629*64)/2); % TODO fix it
      my_string = sprintf('%04.2f,%04.2f,%04.2f', data_pos(jj,1), data_pos(jj,2), data_pos(jj,3));
%       my_string = sprintf('%04.2f,%04.2f,%04.2f', data_pos(jj,2), data_pos(jj,3), data_pos(jj,4));

      for ii=1:length(dataindex)
        if dataindex(ii)
          %     format = '%6.4g';
          %     switch ''
          %       case ''
          %     end
          my_data = data_output{ii}.data(jj,1) / data_output{ii}.Normalization;
          my_string = sprintf('%s,%g', my_string, my_data);
        end
      end
      fprintf(fid, [my_string, '\n']);
    end
  case 2
    main_funny_array = handles.data_pos;
    idx = 4;
    for ii=1:length(dataindex)
      if dataindex(ii)
        my_data = data_output{ii}.data(:,1) / data_output{ii}.Normalization;
        main_funny_array(:,idx) = my_data;
        idx = idx + 1;
      end
    end
    assignin('base', 'hard_working_results', main_funny_array);
    
    uisave('main_funny_array');
end

fclose(fid);


% --------------------------------------------------------------------
function fill_PopupMenu(handles, handles_pm, im_name, add_do_not_use)
switch im_name
  case 'master'
    ImageList = arbuz_FindImage(handles.hh, 'master', '', '', {'Name'});
  otherwise
    ImageList = arbuz_FindImage(handles.hh, 'master', 'Name', im_name, {'SlaveList'});
    ImageList = arbuz_FindImage(handles.hh, ImageList{1}.SlaveList, 'ImageType', '3DMASK', {'Name'});
    %     find_objects = arbuz_FindImage(handles.hh, ImageList{1}.SlaveList, ...
    %       'ImageType', '3DMASK', {'Name'});
  %     find_objects = arbuz_FindImage(handles.hh, ImageList{1}.SlaveList, ...
    %       'ImageType', '3DMASK', {'Name'});
end

if add_do_not_use, str{1} = 'Do not use'; else str={}; end

if ~isempty(ImageList)
  for ii=1:length(ImageList)
    str{end+1}=ImageList{ii}.Name;
  end
end
set(handles_pm, 'String', str, 'value', 1);

% --------------------------------------------------------------------
function eLog_Callback(hObject, eventdata, handles)


% --------------------------------------------------------------------
function eLog_CreateFcn(hObject, eventdata, handles)
% hObject    handle to eLog (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --------------------------------------------------------------------
function pbExportImages_Callback(hObject, eventdata, handles)
data_output = handles.data_output;

ncol  = 0;

dataindex = false(length(data_output), 1);

for ii=1:length(data_output)
  if ~isempty(data_output{ii}.data)
    dataindex(ii) = true;
    ncol = ncol + 1;
    ndata = length(data_output{ii}.data);
  end
end 

output = [];

ncol = 1;
for ii=1:length(dataindex)
  if dataindex(ii)
    newimage = zeros(size(handles.data_mask));
    newimage(handles.data_mask) = data_output{ii}.data(:,1) / data_output{ii}.Normalization;
    output.(handles.set{ii}.name) = newimage;
    ncol = ncol + 1;
  end
end

% data_location = handles.data_location;
assignin('base', 'results', output); %, data_location);
disp('Data are stored in results.');

% --------------------------------------------------------------------
function pmPET_Callback(hObject, eventdata, handles)
items = handles.pmPET.String;
item = items{handles.pmPET.Value};

fill_PopupMenu(handles, handles.pmPETMask, item, true)


% --------------------------------------------------------------------
function pmPET_CreateFcn(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pmPETMask_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pmPETMask_CreateFcn(hObject, eventdata, handles)

