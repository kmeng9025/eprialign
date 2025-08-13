function varargout = IMRT_PET_RPT(varargin)
% IMRT_PET_RPT MATLAB code for IMRT_PET_RPT.fig
%      IMRT_PET_RPT, by itself, creates a new IMRT_PET_RPT or raises the existing
%      singleton*.
%
%      H = IMRT_PET_RPT returns the handle to a new IMRT_PET_RPT or the handle to
%      the existing singleton*.
%
%      IMRT_PET_RPT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMRT_PET_RPT.M with the given input arguments.
%
%      IMRT_PET_RPT('Property','Value',...) creates a new IMRT_PET_RPT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before IMRT_PET_RPT_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to IMRT_PET_RPT_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help IMRT_PET_RPT

% Last Modified by GUIDE v2.5 10-Aug-2023 17:51:54

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @IMRT_PET_RPT_OpeningFcn, ...
  'gui_OutputFcn',  @IMRT_PET_RPT_OutputFcn, ...
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
function IMRT_PET_RPT_OpeningFcn(hObject, ~, handles, varargin)
if ~isempty(varargin) && strcmp(varargin{1}, 'ForceRedraw'), return; end
handles.output = hObject;
handles.hGUI   = varargin{1};

handles.color.correct = [1,1,1];
handles.color.incorrect = [1,0.85,0.85];
handles.color.inproject = [0.8,0.8,0.9];

set(handles.pmSelectExperiment, 'String', {'Select Experiment ...',...
  'pO2 -> HYPOXIA', ... % !
  'PET -> pO2 -> 10 torr', .... 
  'PET -> TRESHOLD*mean(tumor SUV)', .... % !
  'PET -> SUV -> TRESHOLD'});
% loader name, master, slave, type, extension, need to load
file_load = {...
  'MRIax', 'MRI_axial', '', 'MRI', '*.img', true;...
  'MRIsg','MRI_saggital', '','MRI','*.img', false;...
  'CT','CT', '','DICOM3D','*.dcm', true;...
  'EPRfid','FID', '','3DEPRI','*.mat', true;...
  'EPRpO2','pO2_2', '','PO2_pEPRI','*.mat', true;...
  'EPRAmp','Amp', '','AMP_pEPRI','*.mat', false;...
  'PET','PET', '','BIN','*.dat', true;...
  'SE','DCE', '','IDL','*.', true;...
  'KV','DCE', 'Ktrans','IDL','*.', true;...
  'Ve','DCE', 'Ve','IDL','*.', true;...
  'EPRpO2other','pO2', '','PO2_pEPRI','*.mat', true;...
  %  'EPRpO2other2','pO2_3','PO2_pEPRI','*.mat', false...
  };
handles.Files = file_load(:,1);

edit_width = 75;
edit_height = 2.9;
file_block_heght = 3.0;
proc_block_height = 1.9;
button_height = 1.8;
button_height2 = 2.2;
button_width  = 5.0;
button_width2 = 5.4;
file_control_callback = @(hObject,eventdata)IMRT_PET_RPT('Action_Callback',hObject,eventdata,guidata(hObject));
process_control_callback = @(hObject,eventdata)IMRT_PET_RPT('ActionProcess_Callback',hObject,eventdata,guidata(hObject));

root_dir = fileparts(which('ArbuzGUI'));
open_file = imread(fullfile(root_dir, 'images', 'open_file.JPG'));
run_script = imread(fullfile(root_dir, 'images', 'run_script.JPG'));

pos = handles.panFiles.Position;
for ii=1:length(handles.Files)
  vertical_position = pos(4) - ii*file_block_heght - 0.8;
  vertical_position_offset = vertical_position + 0.1*file_block_heght;
  if contains(handles.Files{ii}, 'EPRpO2other')
    vertical_position = vertical_position - 2*file_block_heght;
    handles.(handles.Files{ii}).edit   = uicontrol(handles.panFiles, 'style', 'listbox',...
      'units','characters','position',[pos(3)-edit_width-2.0, vertical_position, edit_width, edit_height*3],...
      'value', 2,'max',5);
  else
    handles.(handles.Files{ii}).edit   = uicontrol(handles.panFiles, 'style', 'edit',...
      'units','characters','position',[pos(3)-edit_width-2.0, vertical_position, edit_width, edit_height],...
      'value', 2,'max',5);
  end
  handles.(handles.Files{ii}).browse = uicontrol(handles.panFiles, 'style', 'pushbutton',...
    'units','characters','position',[20, vertical_position_offset, button_width2, button_height2],...
    'callback',file_control_callback, 'CData', open_file);
  handles.(handles.Files{ii}).load   = uicontrol(handles.panFiles, 'style', 'pushbutton',...
    'units','characters','position',[25.5, vertical_position_offset, button_width2, button_height2],...
    'callback',file_control_callback, 'CData', run_script);
  handles.(handles.Files{ii}).checkbox   = uicontrol(handles.panFiles, 'style', 'checkbox',...
    'units','characters','position',[1.0, vertical_position_offset, 18.0, button_height2],...
    'string', handles.Files{ii},'callback',file_control_callback);

  handles.(handles.Files{ii}).id   = file_load{ii,2};
  handles.(handles.Files{ii}).idslave = file_load{ii,3};
  handles.(handles.Files{ii}).type = file_load{ii,4};
  handles.(handles.Files{ii}).exts = file_load{ii,5};
  handles.(handles.Files{ii}).need_to_load = file_load{ii,6};
  handles.(handles.Files{ii}).inproject = false;
end

processing_steps = {...
  'ProcessEPRImages','Reconstruct EPR images';
  'CreateRegistrationSequence','Create registration sequence';
  'UpdateLoadedTransformation', 'Update loaded images transformation';
  'SegmentMRIaxFiducials','Segment MRI axial image';
  'SegmentMRIsagFiducials','Segment MRI saggital image';
  'SegmentEPRfiducials','Segment EPR fiducial image';
  'SegmentPET','Segment PET';
  'SegmentCTfiducials','Segment CT image';
  'RegisterMRI','Register MRI';
  'RegisterPETDCE','Register PET/DCE';
  'RegisterCT','Register CT';
  'SegmentTumor','Segment tumor from MRI (dummy)';
  'Visualization', 'Visualization';
  'PreparePETData','PET -> SUV/PO2';
  'PrepareImageData','Prepare data for IMRT';
  'WholeFieldPlanning','Plan whole field radiation dose';
  'BoostFieldDosePlanning','Plan boost/aboost radiation dose';
  'PlanTreatment','Prepare radiation block';
  'GeneralStatistics', 'General statistics';
  'CoverageMap', 'Coverage Map (long)'
  };

%   'PrepareTumorMask','Condition hypoxia mask';

handles.Processings = processing_steps(:,1);

pos = handles.panProcessing.Position;
for ii=1:length(handles.Processings)
  vertical_position = pos(4) - ii*proc_block_height - 3.7;
  handles.(handles.Processings{ii}).checkbox  = uicontrol(handles.panProcessing, 'style', 'checkbox',...
    'units','characters','position',[1.0, vertical_position, pos(3)-12, 2.0],...
    'string', processing_steps{ii,2},'callback',process_control_callback);
  handles.(handles.Processings{ii}).pb_options = uicontrol(handles.panProcessing, 'style', 'pushbutton',...
    'units','characters','position',[pos(3)-3.0-10.0-3*button_width, vertical_position, 2*button_width, button_height],...
    'callback',process_control_callback,'String','options');
  handles.(handles.Processings{ii}).pb_run = uicontrol(handles.panProcessing, 'style', 'pushbutton',...
    'units','characters','position',[pos(3)-2.0-10.0-button_width, vertical_position, button_width, button_height],...
    'callback',process_control_callback, 'CData', run_script);
  handles.(handles.Processings{ii}).pb_report = uicontrol(handles.panProcessing, 'style', 'pushbutton',...
    'units','characters','position',[pos(3)-1.0-10.0, vertical_position, 10.0, button_height],...
    'callback',process_control_callback,'String','report');
  handles.(handles.Processings{ii}).options=[];
end
handles.SegmentMRIaxFiducials.options = struct('first_slice', 1, 'last_slice', 100, 'image_threshold', 0.1, 'fiducial_number', 4, 'fiducials_threshold', 0.25,...
  'largest_noise_object', 7, 'dilate_radius', 2, 'extract_outline', true, 'look_for_expansion', false);
handles.SegmentMRIsagFiducials.options = struct('first_slice', 1, 'last_slice', 1000, 'image_threshold', 0.08, 'fiducial_number', 4, 'fiducials_threshold', 0.35,...
  'largest_noise_object', 7, 'dilate_radius', 2, 'extract_outline', true);
handles.SegmentEPRfiducials.options = struct('first_slice', 1, 'last_slice', 1000,'image_threshold', 0.25, 'fiducial_number', 4, 'fiducials_threshold', 0.25,...
  'largest_noise_object', 65, 'dilate_radius', 2, 'extract_outline', false,...
  'look_for_expansion', false);
handles.SegmentPET.options = struct('image_threshold', 0.09);
handles.SegmentCTfiducials.options = struct('first_slice', 140, 'last_slice', 180,'fiducials_number',4,...
  'fiducials_voxels',200,'cast_density_min', 3500, 'animal_density_min',400,'animal_density_max',2000,...
  'noise_density_max', 450);
handles.RegisterMRI.options = struct('fiducial_number', 4);
handles.RegisterCT.options = struct('fiducial_number', 4);
handles.RegisterPETDCE.options = [];
handles.PreparePETData.options = struct('PreparationTime', datestr(now), 'InjectionTime', datestr(now), 'PostInjectionTime', datestr(now), ...
  'AcquisitionStartTime', datestr(now), 'AcquisitionEndTime', datestr(now), 'SyringeDose', 0, 'LeftoverDose', 0, 'MouseWeight', 0, ...
  'export_data', 0);
handles.PrepareImageData.options = struct('use_mask', 'Tumor', 'use_mri_mask', 1, 'threshold', 1);
handles.WholeFieldPlanning.options = struct('prescription_Gy', 49.9, 'Field_size_mm', 35, ...
  'noise_density_max', 450, 'bone_segmentation_retrofix', 0);

handles.BoostFieldDosePlanning.options = struct('boost_1_aboost_2', 1, 'beams', 2, 'prescription_Gy', 13, ...
  'boost_margin', 1.2, 'noise_density_max', 450, 'bone_segmentation_retrofix', 0);
handles.PlanTreatment.options = struct('boost_1_aboost_2', 1, 'beams', 2, 'prescription_Gy', 13, ...
  'boost_margin', 1.2, 'antiboost_margin', 0.6, 'Plug_size', 16, 'noise_density_max', 450, ...
  'antiboost_algorithm', 0);

set(handles.eProjectPath, 'String', fileparts(arbuz_get(handles.hGUI, 'FileName')));

handles.PrepData.Treatment_inCT = [];
handles.PO2_inCT = [];
handles.monitoring.anesth_iso = [];
handles.monitoring.temperature = [];
handles.monitoring.bpm = [];
handles.monitoring.Qvalue = [];
handles.monitoring.signal = [];

handles.MRIf4Pars = [];
handles.MRIf4Pars.stages = [0.15, 0.25, 0.7];
handles.MRIf4Pars.defaults = 'MRI';
handles.MRIf4Pars.extentz = [-4 6];
handles.MRIf4Pars.figure = 1000;
handles.MRIf4Pars.erode_layers = [0,0,1];
handles.MRIf4Pars.fit_threshold = 0.55;
handles.MRIf4Pars.pause = false;

handles.EPRf4Pars = [];
handles.EPRf4Pars.stages = [0.2, 0.4, 0.8];
handles.EPRf4Pars.defaults = 'EPR';
handles.EPRf4Pars.extentz = [-40 60];
handles.EPRf4Pars.figure = 1500;
handles.EPRf4Pars.erode_layers = [0,0,1];

handles.CTf4Pars = [];
handles.CTf4Pars.stages = [0.01, 0.035, 0.07];
handles.CTf4Pars.defaults = 'CT';
handles.CTf4Pars.extentz = [-5 8];
handles.CTf4Pars.figure = 1800;
handles.CTf4Pars.erode_layers = [0,0,1];

handles.location.Matlab = 'd:\CenterMATLAB';
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = IMRT_PET_RPT_OutputFcn(~, ~, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pmSelectExperiment_Callback(~, ~, handles)

Status = arbuz_get(handles.hGUI, 'Status');
Status('IMRTprotocol') = handles.pmSelectExperiment.Value;
arbuz_set(handles.hGUI, 'Status', Status);

switch handles.pmSelectExperiment.Value
  case 2 % pO2 -> HYPOXIA
    AddMessage(handles, '#set_options','PO2-HYPOXIA', true);
    handles.PrepareImageData.options.threshold = 100;
  case 3 % PET -> pO2 -> 10 torr
    AddMessage(handles, '#set_options','PET-PO2-HYPOXIA', true);
  case 4 % PET -> SUV -> MAX/TRESHOLD
    AddMessage(handles, '#set_options','PET-SUV-MAX', true);
    handles.PrepareImageData.options.threshold = 1.4;
  case 5 % PET -> SUV -> TRESHOLD
    AddMessage(handles, '#set_options','PET-SUV-MN', true);
end

handles.LoadedParameters.Protocol = handles.pmSelectExperiment.Value;
handles = LoadProject(handles);
PrepareThePlan(handles);

% --------------------------------------------------------------------
function pbProjectDirectory_Callback(~, ~, handles)
[PathName] = uigetdir(uncell(get(handles.eProjectPath, 'String')));
if ~isequal(PathName, 0)
  set(handles.eProjectPath, 'String', PathName);
  PrepareThePlan(handles);
end

% --------------------------------------------------------------------
function pbLoadProject_Callback(~, ~, handles)
set(handles.eProjectPath, 'String', fileparts(arbuz_get(handles.hGUI, 'FileName')));

handles = LoadProject(handles);
handles.pmSelectExperiment.Value = handles.LoadedParameters.Protocol;

PrepareThePlan(handles);

% --------------------------------------------------------------------
function handles = LoadProject(handles)

% clear out the prepared data
handles.PrepData.Treatment_inCT = [];
handles.PrepData.PO2_inCT = [];

[handles.LoadedParameters.Protocol,Status] = arbuz_get_status(handles.hGUI, 'IMRTprotocol', 1);
switch handles.LoadedParameters.Protocol
  case 2 % PET
    [handles.LoadedParameters.normoxia_dose] = arbuz_get_status(handles.hGUI, 'IMRTnormoxia_dose', 12);
    [handles.LoadedParameters.hypoxia_dose, Status] = arbuz_get_status(handles.hGUI, 'IMRThypoxia_dose', 35);
    AddMessage(handles, '#set_options','PETv1.0', true);
end

% update options
for ii=1:length(handles.Processings)
  options = handles.(handles.Processings{ii}).options;
  if ~isempty(options)
    fields = fieldnames(options);
    if Status.isKey(handles.Processings{ii})
      saved_options = Status(handles.Processings{ii});
      for jj=1:length(fields)
        if isfield(saved_options, fields{jj}) && isfield(options, fields{jj})
          handles.(handles.Processings{ii}).options.(fields{jj}) = saved_options.(fields{jj});
        end
      end
    end
  end
end

% --------------------------------------------------------------------
function Action_Callback(hObject, eventdata, handles)
for ii=1:length(handles.Files)
  cset = handles.(handles.Files{ii});
  if hObject == cset.load
    if contains(handles.Files{ii}, 'EPRpO2other')
      % load images different from already loaded
      flist = get(cset.edit, 'String');
      for jj=1:length(flist)
        output_list = arbuz_FindImage(handles.hGUI, 'master', 'FileName', flist{jj}, {});
        if isempty(output_list)
          load_image(handles.hGUI, flist{jj}, [cset.id,'_o',num2str(jj)], cset.idslave, cset.type);
        end
      end
    else
      load_image(handles.hGUI, get(cset.edit, 'String'), cset.id, cset.idslave, cset.type);
    end
    ReadFileName(handles, handles.Files{ii});
    handles.(handles.Files{ii}).inproject = true;
    if ~isempty(eventdata)
      ActionProcess_Callback(handles.UpdateLoadedTransformation.pb_run, eventdata, handles);
    end
    return;
  elseif hObject == cset.browse
    if contains(handles.Files{ii}, 'EPRpO2other')
      flist = get(cset.edit, 'String');
      if isempty(flist)
        cset = handles.('EPRpO2');
        open_dir = fileparts(get(cset.edit, 'String'));
      elseif ischar(flist)
        open_dir = flist;
      else
        open_dir = flist{1};
      end
      if exist(open_dir, 'file') ~= 7
        open_dir = handles.eProjectPath.String;
        if ~exist(open_dir, 'file') ~= 7
        end
      end
      [fname, folder] = uigetfile({cset.exts, ['Files (',cset.exts,')']}, 'Select a file', open_dir, 'MultiSelect', 'on');
      if ~isequal(fname, 0)
        flist = cell(length(fname), 1);
        for jj=1:length(fname), flist{jj} = fullfile(folder, fname{jj}); end
        set(cset.edit, 'String', fullfile(folder, fname));
      end
    else
      open_dir = fileparts(get(cset.edit, 'String'));
      if exist(open_dir, 'file') ~= 7
        open_dir = handles.eProjectPath.String;
        if ~exist(open_dir, 'file') ~= 7
        end
      end
      [fname, folder] = uigetfile({cset.exts, ['Files (',cset.exts,')']}, 'Select a file', open_dir, 'MultiSelect', 'off');
      if ~isequal(fname, 0)
        set(cset.edit, 'String', fullfile(folder, fname));
      end
    end
    return;
  elseif hObject == cset.checkbox
    handles.UpdateLoadedTransformation.checkbox.Value = 1;
    return;
  end
end
guidata(hObject, handles);
if contains(class(eventdata), 'ActionData')
  PrepareThePlan(handles);
end

% --------------------------------------------------------------------
function ActionProcess_Callback(hObject, ~, handles)
for ii=1:length(handles.Processings)
  cset = handles.(handles.Processings{ii});
  if hObject == cset.checkbox
    break;
  elseif hObject == cset.pb_options
    if ~isempty(cset.options)
      names = fieldnames(cset.options);
      prompt = cell(length(names),1); values = cell(length(names),1);
      for jj=1:length(names)
        prompt{jj}=names{jj};
        values{jj} = num2str(cset.options.(names{jj}));
      end
      res=inputdlg(prompt, 'Set options',1,values);
      if ~isempty(res)
        for jj=1:length(names)
          result = str2double(res{jj});
          the_option = handles.(handles.Processings{ii}).options.(names{jj});
          if(~isnan(result))
            handles.(handles.Processings{ii}).options.(names{jj}) = result;
          elseif ischar(the_option)
            handles.(handles.Processings{ii}).options.(names{jj}) = res{jj};
          end
        end
      end
    end
    break;
  elseif hObject == cset.pb_report
    fname = fullfile(handles.eProjectPath.String, handles.eFolder.String, handles.Processings{ii});
    flist = dir([fname,'*.png']);
    for jj=1:length(flist)
      fname = fullfile(flist(jj).folder, flist(jj).name);
      im = imread(fname);
      figure('Name',fname,'NumberTitle','off'); image(im);axis image; axis off;
      set(gca, 'Position',[0.02,0.02,0.96,0.96])
    end
    flist = dir([fname,'*.fig']);
    for jj=1:length(flist)
      fname = fullfile(flist(jj).folder, flist(jj).name);
      open(fname);
    end
    break;
  elseif hObject == cset.pb_run
    for jj=1:length(handles.Files)
      set(handles.(handles.Files{jj}).checkbox, 'value', false);
    end
    for jj=1:length(handles.Processings)
      set(handles.(handles.Processings{jj}).checkbox, 'value', false);
    end
    set(handles.(handles.Processings{ii}).checkbox, 'value', true);
    pbRunAll_Callback(hObject, [], handles);
    return;
  end
end

Status = arbuz_get(handles.hGUI, 'Status');
for ii=1:length(handles.Processings)
  Status(handles.Processings{ii}) = handles.(handles.Processings{ii}).options;
end
arbuz_set(handles.hGUI, 'Status', Status);


guidata(hObject, handles);

% --------------------------------------------------------------------
function res = FindFileName(PathName, file_type)
res = '';
EPRPathName = MRI2EPRpath(PathName);
switch file_type
  case 'MRIax'
    res1 = findMRIFile(PathName, 'axial');
    for ii=1:length(res1)
      if exist(res1{ii}, 'file') == 2 % file exists
        res = res1{ii};
        break;
      end
    end
  case 'MRIsg'
    res1 = findMRIFile(PathName, 'sagittal');
    for ii=1:length(res1)
      if exist(res1{ii}, 'file') == 2 % file exists
        res = res1{ii};
        break;
      end
    end
  case 'CT', [~, res] = is_data_file(PathName, '*.dcm', 1);
  case 'PET', [~, res] = is_data_file(PathName, 'PET*.dat', 1);
  case 'SE', [~, res] = is_data_file(PathName, 'SE21.', 1);
  case 'KV', [~, res] = is_data_file(PathName, 'KVmaps2.', 1);
  case 'Ve', [~, res] = is_data_file(PathName, 'KVmaps2.', 1);
  case 'EPRfid', [~, res] = is_data_file(PathName, '*image3D*.mat',1);
  case {'EPRpO2', 'EPRAmp'}, [~, res] = is_data_file(PathName, 'p*imageT1*.mat', 2);
  case 'EPRpO2other', [~, res] = is_data_file(PathName, 'p*imageT1*.mat', 1);
  case 'EPRpO2other2', [~, res] = is_data_file(PathName, 'p*imageT1*.mat', 3);
end

% --------------------------------------------------------------------
function res = ReadFileName(handles, file_type)
res = '';
switch file_type
  case 'MRIax'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'MRI', {'FileName'});
    if length(im_list)==1
      res = im_list{1}.FileName;
    else
      im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', 'ax', {'FileName'});
      if ~isempty(im_list), res = im_list{1}.FileName; end
    end
  case 'MRIsg'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'MRI', {'FileName'});
    im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', 'sag', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
  case 'CT'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'DICOM3D', {});
    im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', 'CT', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
  case 'PET'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'BIN', {});
    im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', 'PET', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
  case 'EPRfid'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', '3DEPRI', {'FileName'});
    %     im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', 'fid-auto', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
  case 'EPRpO2'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'PO2_pEPRI', {'FileName'});
    im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', '_2', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
  case 'EPRAmp'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'AMP_pEPRI', {'FileName'});
    %     im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', 'AMP', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
  case 'EPRpO2other'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'PO2_pEPRI', {'FileName'});
    res = cell(length(im_list), 1);
    for ii=1:length(im_list), res{ii} = im_list{ii}.FileName; end
  case 'EPRpO2other2'
    im_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'PO2_pEPRI', {'FileName'});
    im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', '_3', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
end

% --------------------------------------------------------------------
function inp = uncell(inp)
if iscell(inp), inp = inp{1}; end

% --------------------------------------------------------------------
function folders = dirlist(fpath)
if exist(fpath, 'file') ~= 7, folders = []; return; end
files = dir(fpath); files(1:2)=[];
folders = files([files.isdir]);

% --------------------------------------------------------------------
function [isMRI, MRI_file_path] = isMRIfolder(fpath, MRI_type)
isMRI = false; MRI_file_path = {};
folders = dirlist(fpath);
fnames = {folders(:).name};
type = {};
for ii=1:10
  type{ii} = '';
  if any(cellfun( @(x) strcmp(x,num2str(ii)), fnames ))
    MRI_path = fullfile(fpath, num2str(ii), 'method');
    text = fileread(MRI_path);
    if strfind(text, 'sagittal'), type{ii} = 'sagittal'; end
    if strfind(text, 'axial'), type{ii} = 'axial'; end
  end
end
for ii=10:-1:1
  if strfind(type{ii}, MRI_type)
    isMRI = true;
    MRI_file_path{1} = fullfile(fpath, num2str(ii), 'pdata', '1', '2dseq.img');
    MRI_file_path{2} = fullfile(fpath, num2str(ii), 'pdata', '1', '2dseq.');
    return;
  end
end
MRI_file_path{1} = fpath;

% --------------------------------------------------------------------
function MRI_file = findMRIFile(fpath, MRI_type)
folders = dirlist(fpath);
MRI_file = fpath;
for ii=1:length(folders)
  MRI_folder = fullfile(fpath, folders(ii).name);
  [is, fname] = isMRIfolder(MRI_folder, MRI_type);
  if is, MRI_file = fname; break; end
end
if ~iscell(MRI_file), MRI_file = {MRI_file}; end

% --------------------------------------------------------------------
function [is_found, out_path] = is_data_file(fpath, pattern, n)
[is_found, out_path] = isfolder(fpath, pattern);
out_path = out_path{n};

% --------------------------------------------------------------------
function [is_found, out_path] = isfolder(fpath, pattern)
[is_found, out_path] = are_files(fpath, pattern);
if ~is_found
    folders = dir(fpath);
    if isempty(folders), return; end
    folders = folders([folders.isdir]);
    folders = folders(~ismember({folders(:).name},{'.','..'}));
    for ii=1:length(folders)
        [is_found, out_path] = isfolder(fullfile(fpath, folders(ii).name), pattern);
        if is_found, break; end
    end
end

% --------------------------------------------------------------------
function [is_found, out_path] = are_files(fpath, pattern)
is_found = false; out_path = {fpath};
files = dir(fullfile(fpath, pattern));
if isempty(files), return; end
files = files(~[files.isdir]);
is_found = true;
for ii=1:length(files)
  out_path{ii} = fullfile(fpath, files(ii).name);
end

% --------------------------------------------------------------------
function load_image(hGUI, FileName, image_name, slave_image_name, image_type)
set(hGUI,'Pointer','watch');drawnow
parameters = [];

image_to_load = image_name;
if ~isempty(slave_image_name), image_to_load = slave_image_name; end

if contains(image_to_load, 'PET')
  [fp, fn] = fileparts(FileName);
  fd = fullfile(fp, [fn, '.arbuz']);
  if ~exist(fd, "file") ~= 2
    fid = fopen(fd, 'wt');
    fprintf(fid, '[T1]\nresolution = [0.5 0.5 0.5]\noffset = [0.0 0.0 0.0]\n\n');
    fprintf(fid, '[data]\nformat = float32\ndims = [72 72 52]\n\n');
    fclose(fid);
  end
elseif contains(image_to_load, 'Ktrans')
  % [fp, fn] = fileparts(new_image.FileName);
  % fd = fullfile(fp, [fn, '.arbuz']);
  % if ~exist(fd, "file") ~= 2
  %   fid = fopen(fd, 'wt');
  %   fprintf(fid, '[T1]\nresolution = [0.5 0.5 0.5]\noffset = [0.0 0.0 0.0]\n\n');
  %   fprintf(fid, '[data]\nformat = float32\ndims = [72 72 52]\n\n');
  %   fclose(fid);
  % end
  parameters.slice = 1;
elseif contains(image_to_load, 'Ve')
  parameters.slice = 2;
end

new_image = create_new_image(image_to_load, image_type, []);
new_image.FileName = FileName;
new_image.Name = image_to_load;

[new_image.data, new_image.data_info] = arbuz_LoadImage(FileName, image_type, parameters);
if isempty(new_image.data)
  if contains(new_image.ImageType, 'MRI')
    [new_image.data, new_image.data_info] = arbuz_LoadImage(new_image.FileName, 'DICOM3D');
  end
end
new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
new_image.Aprime = eye(4);
set(hGUI,'Pointer','arrow');drawnow

if isempty(slave_image_name)
  new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
  arbuz_AddImage(hGUI, new_image);
  arbuz_StageToTransformation(hGUI, 'T2');
  output_list = arbuz_FindImage(hGUI, 'master', 'Name', new_image.Name, {'Name', 'Anative'});
  arbuz_SetTransformation(hGUI, 'T1', output_list{1}.Name, output_list{1}.Anative);
else
  new_image.Anative = eye(4);
  output_list = arbuz_FindImage(hGUI, 'master', 'Name', image_name, {'Name', 'Anative'});
  if ~isempty(output_list)
    arbuz_AddImage(hGUI, new_image, output_list{1}.Image);
  else
    CheckData(output_list, 'Master image was not found')
  end
end
arbuz_UpdateInterface(hGUI);

% --------------------------------------------------------------------
function  EPRPathName = MRI2EPRpath(PathName)
date_path = epr_DateFromPath(PathName);
try
  date = datenum(['20', date_path(1:2),'-',date_path(3:4), '-', date_path(5:6)]);
  MRIPathName = epr_PathFromDate(date, 'imagnet', '');
catch
  MRIPathName = '';
end

mrifolder = PathName(length(MRIPathName)+1:end);
mrifolder(mrifolder == filesep)='';
try
  EPRPathName = epr_PathFromDate(date, 'pulse250', '');
catch
  EPRPathName = '';
end

if ~isempty(mrifolder)
  files = dirlist(EPRPathName);
  idx = zeros(length(files), 1);
  for ii=1:length(files)
    idx(ii) = strdist(files(ii).name, mrifolder);
  end
  [~,myidx] = min(idx);
  if ~isempty(files)
    EPRPathName = epr_PathFromDate(date, 'pulse250', files(myidx).name);
  end
end

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
function pbRunAll_Callback(hObject, ~, handles)
if handles.pmSelectExperiment.Value == 1
  warning('Experiment is not selected.');
  return;
end

project_state = arbuz_get(handles.hGUI, 'state');

handles.panProcessing.ForegroundColor = 'red';
handles.panProcessing.Title = 'Processing - active';
set(handles.panProcessing.Children, 'Enable', 'off');
set(handles.panFiles.Children, 'Enable', 'off');
set(handles.uipanel1.Children, 'Enable', 'off');
pause(0.1);

try
  IMRTfolder = handles.eFolder.String;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % load all files
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for ii=1:length(handles.Files)
    if handles.(handles.Files{ii}).checkbox.Value == 1
      Action_Callback(handles.(handles.Files{ii}).load, [], guidata(hObject));
      handles.UpdateLoadedTransformation.checkbox.Value = true;
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % process EPR images
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.ProcessEPRImages.checkbox.Value
    arbuz_ShowMessage(handles.hGUI, 'Processing EPR images.');
    PathName = handles.eProjectPath.String;
    EPRPathName = MRI2EPRpath(PathName);

    % check if fiducials image is available
    [~, res] = is_data_file(PathName, '*image3D*.tdms');

    % check if po2 images are available
    [~, res] = isfolder(PathName, '*imageT1*.tdms');
    for ii=1:length(res)
      [~, fn]=fileparts(res{ii});
      if exist(fullfile(PathName, ['p',fn,'.mat']), 'file') ~= 2 % process file
        ProcessEPRImage(handles, PathName, res{ii}, 'pO2');
      end
    end
    %       if exist(res, 'file') ~= 2 % process file
    %         res = findEPRFile(EPRPathName, 'pO2raw');
    %         if ~isempty(res)
    %           ProcessEPRImage(PathName, res, 'pO2');
    %         end
    %       end

  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % create registration sequence
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.CreateRegistrationSequence.checkbox.Value
    arbuz_ShowMessage(handles.hGUI, 'Create transformations.');
    arbuz_AddTransformation(handles.hGUI, 'T1');
    arbuz_AddTransformation(handles.hGUI, 'M1');
    arbuz_AddTransformation(handles.hGUI, 'T-PET');
    arbuz_AddTransformation(handles.hGUI, 'T-CT');
    arbuz_AddTransformation(handles.hGUI, 'T2');
    arbuz_AddSequence(handles.hGUI, 'S1');

    arbuz_ShowMessage(handles.hGUI, 'Setting T1 transformation to images.');
    output_list = arbuz_FindImage(handles.hGUI, 'master', '', '', {'Name', 'Anative'});
    for ii=1:length(output_list)
      arbuz_SetTransformation(handles.hGUI, 'T1', output_list{ii}.Name, output_list{ii}.Anative);
    end

    % unselect all images
    arbuz_SetImage(handles.hGUI, output_list, 'Selected', 0);

    arbuz_ShowMessage(handles.hGUI, 'Creating sequence ...');
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
    Sequences{1}.Sequence{4}.Description = 'ct->MRI';

    arbuz_ShowMessage(handles.hGUI, 'Creating MRI to EPRI.');
    Sequences{1}.Sequence{5}.Name = 'T2';
    Sequences{1}.Sequence{5}.Description = 'MRI->EPRI';

    arbuz_set(handles.hGUI, 'Sequences', Sequences);
    arbuz_set(handles.hGUI, 'ACTIVETRANSFORMATION', 'T2');
    arbuz_set(handles.hGUI, 'WATCHTRANSFORMATION', 'T2');
  end

  image_CT = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'DICOM3D', {'slavelist'});
  image_CT = arbuz_FindImage(handles.hGUI, image_CT, 'InName', 'CT', {'slavelist'});
  image_MRIax = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'MRI', {});
  image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, 'InName', '_ax', {'slavelist','Anative'});
  image_PO2 = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'PO2_pEPRI', {});
  image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, 'InName', '_2', {'slavelist'});
  image_FID = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', '3DEPRI', {});
  image_FID = arbuz_FindImage(handles.hGUI, image_FID, 'Name', 'FID', {'slavelist'});
  image_PET = arbuz_FindImage(handles.hGUI, 'master', 'Name', 'PET', {'slavelist'});

  Status = arbuz_get(handles.hGUI, 'Status');

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % load T1 transformation for those files loaded
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.UpdateLoadedTransformation.checkbox.Value
    arbuz_ShowMessage(handles.hGUI, 'Setting T1 transformation to images.');
    arbuz_StageToTransformation(handles.hGUI, 'T2');
    output_list = arbuz_FindImage(handles.hGUI, 'master', '', '', {'Name', 'Anative', 'Acurrent'});
    for ii=1:length(output_list)
      if isequal(output_list{ii}.Acurrent, eye(4)) && ~isequal(output_list{ii}.Anative, eye(4))
        arbuz_SetTransformation(handles.hGUI, 'T1', output_list{ii}.Name, output_list{ii}.Anative);
      end
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment MRI fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentMRIaxFiducials.checkbox.Value
    AddProcessingOptions(handles, 'SegmentMRIaxFiducials');
    image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist','data'});
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentMRIaxFiducials');

    for ii=1:length(image_MRIax)
      opts = handles.SegmentMRIaxFiducials.options;

      [outline, fiducials] = arbuz_fiducial_segmentation(image_MRIax{ii}.data, opts);

      presentation = outline + 2.5*fiducials;
      slice_range = opts.first_slice:min(opts.last_slice,size(outline,3));
      fig_opts.legend = 'blue: animal; green: fiducials; red: not assigned';
      fig_opts.show_min = 100;
      fig_opts.show_max = 12000;
      figN = imrt_show_segmentation(1, image_MRIax{ii}.data, presentation, slice_range, fig_opts);
      set(figN, 'Position', get(0, 'Screensize'));
      epr_mkdir(fileparts([fname,'1.png']));
      saveas(figN, [fname,'1.png']);
      delete(figN);

      new_image = create_new_image('','3DMASK',[]);
      new_image.data = outline;
      new_image.Name = 'mri-outline-auto';
      arbuz_AddImage(handles.hGUI, new_image, image_MRIax{ii}.Image);
      new_image.data = fiducials;
      new_image.Name = 'mri-fid-auto';
      arbuz_AddImage(handles.hGUI, new_image, image_MRIax{ii}.Image);
    end

    % fit fiducials
    %     image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist'});
    %     output_list = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    %     output_list = arbuz_FindImage(handles.hGUI, output_list, 'InName', 'mri-fid-auto', {'data','Anative'});
    %
    %     [AMR, A2MR, rot_f4r, rot_f4r1] = f4model(output_list{1}.data, output_list{1}.Anative, struct('stages',[0.15 0.25], 'defaults', 'MRI', 'extentz', [-4 +6], 'guess', [0,0,0,0,0],'figure',100,'figure_filename',fname));
    %     res = cell(1,4);
    %     e1 = htransform_vectors(inv(A2MR), rot_f4r);
    %     e2 = htransform_vectors(inv(A2MR), rot_f4r1);
    %     for ii=1:4
    %       res{ii}.ends(1,:) = e1(ii,:);
    %       res{ii}.ends(2,:) = e2(ii,:);
    %     end
    %
    %     new_image = create_new_image('','XYZ',[]);
    %     for ii=1:length(res)
    %       new_image.data = res{ii}.ends;
    %       new_image.Name = sprintf('AxMRIFID%i',ii);
    %       arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    %     end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment MRI fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentMRIsagFiducials.checkbox.Value
    AddProcessingOptions(handles, 'SegmentMRIsagFiducials');
    output_list = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'MRI', {});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'InName', '_sag', {'slavelist','data'});
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentMRIsagFiducials');

    for ii=1:length(output_list)
      opt = handles.SegmentMRIsagFiducials.options;

      [outline, fiducials] = arbuz_fiducial_segmentation(output_list{ii}.data, opt);

      presentation = outline + 2*fiducials;
      slice_range = 1:size(outline,3);
      fig_opts.legend = 'blue: animal; green: fiducials; red: not assigned';
      fig_opts.show_min = 100;
      fig_opts.show_max = 12000;
      figN = imrt_show_segmentation(1, output_list{ii}.data, presentation, slice_range, fig_opts);
      set(figN, 'Position', get(0, 'Screensize'));
      epr_mkdir(fileparts([fname,'1.png']));
      saveas(figN, [fname,'1.png']);
      delete(figN);

      new_image = create_new_image('','3DMASK',[]);
      new_image.data = outline;
      new_image.Name = 'mri-outline-auto';
      arbuz_AddImage(handles.hGUI, new_image, output_list{ii}.Image);
      new_image.data = fiducials;
      new_image.Name = 'mri-fid-auto';
      arbuz_AddImage(handles.hGUI, new_image, output_list{ii}.Image);
    end

    output_list = arbuz_FindImage(handles.hGUI, output_list, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'mri-fid-auto', {'data','Anative'});

    opts = handles.SegmentMRIsagFiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 2;
    opts.figure_filename=fname;
    res = arbuz_fit_fiducials(output_list{1}.data, opts);

    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('SgMRIFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment EPR fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentEPRfiducials.checkbox.Value
    AddProcessingOptions(handles, 'SegmentEPRfiducials');
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist','data', 'Anative'});
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentEPRfiducials');

    options = handles.SegmentEPRfiducials.options;
    resolution = diag(image_FID{1}.Anative); resolution = resolution(1:3);
    [~, eprfiducials] = geom_segment_EPR(image_FID{1}.data, resolution, options);

    % create new 3DMASK for EPR fiducials
    for ii=1:length(image_FID)
      new_image = create_new_image('epr-fid-auto','3DMASK',eprfiducials);
      arbuz_AddImage(handles.hGUI, new_image, image_FID{ii}.Image);
    end

    maxf = max(image_FID{1}.data(:));
    presentation = eprfiducials;
    idx = any(any(presentation, 1),3);
    slice_range = find(idx,1,'first'):find(idx,1,'last');
    fig_opts.legend = 'green: fiducials; red: not assigned';
    fig_opts.show_min = maxf*0.15;
    fig_opts.show_max = maxf;
    fig_opts.slicedir = 2;
    figN = imrt_show_segmentation(1, image_FID{1}.data, presentation, slice_range, fig_opts);
    set(figN, 'Position', get(0, 'Screensize'));
    epr_mkdir(fileparts([fname,'1.png']));
    saveas(figN, [fname,'1.png']);
    delete(figN);

    %  fit fiducials
    %     image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    %     output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    %     output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});
    %
    %     [AEPR, A2, rot_f4r, rot_f4r1] = f4model(output_list{1}.data, output_list{1}.Anative, struct('stages',[0.2 0.6], 'defaults', 'EPR', 'extentz', [-40 +60], 'guess', [0,0,0,0,0],...
    %       'figure',100,'figure_filename',fname));
    %     res = cell(1,4);
    %     e1 = htransform_vectors(inv(A2), rot_f4r);
    %     e2 = htransform_vectors(inv(A2), rot_f4r1);
    %     for ii=1:4
    %       res{ii}.ends(1,:) = e1(ii,:);
    %       res{ii}.ends(2,:) = e2(ii,:);
    %     end
    %
    %     new_image = create_new_image('','XYZ',[]);
    %     for ii=1:length(res)
    %       new_image.data = res{ii}.ends;
    %       new_image.Name = sprintf('AnFID%i',ii);
    %       arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    %     end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment PET fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentPET.checkbox.Value
    AddProcessingOptions(handles, 'SegmentPET');
    output_list = arbuz_FindImage(handles.hGUI, image_PET, 'Name', 'PET', {'slavelist','data'});

    opt = handles.SegmentPET.options;
    outline = output_list{1}.data > max(output_list{1}.data(:) * opt.image_threshold);

    new_image = create_new_image('','3DMASK',[]);
    new_image.data = outline;
    new_image.Name = 'pet-outline-auto';
    arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment CT fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentCTfiducials.checkbox.Value
    AddProcessingOptions(handles, 'SegmentCTfiducials');
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentCTfiducials');
    output_list = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'slavelist','data','Anative'});
    if ~isempty(output_list)
      opt = handles.SegmentCTfiducials.options;
      opt.figure = 1000;
      opt.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentCTfiducials');
      %       [animal, cast, mousebed, fiducials, bone] = arbuz_segment_CT(output_list{1}.data, opt);
      resolution = diag(output_list{1}.Anative); resolution = resolution(1:3);
      [animal, cast, mousebed, fiducials, bone] = geom_segment_CT(output_list{1}.data, resolution, opt);

      new_image = create_new_image('ct-fid-auto','3DMASK',fiducials);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
      new_image.data = cast;
      new_image.Name = 'ct-cast-auto';
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
      new_image.data = animal;
      new_image.Name = 'ct-outline-auto';
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
      new_image.data = mousebed;
      new_image.Name = 'ct-mousebed-auto';
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
      new_image.data = bone;
      new_image.Name = 'ct-bone-auto';
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end

    % fit fiducials
    %     output_list = arbuz_FindImage(handles.hGUI, output_list, '', '', {'slavelist'});
    %     output_list = arbuz_FindImage(handles.hGUI, output_list{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    %     output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'ct-fid-auto', {'data','Anative'});
    %
    %     [ACT, A2CT, rot_f4r, rot_f4r1] = f4model(output_list{1}.data, output_list{1}.Anative, struct('stages', [0.01 0.045], 'defaults', 'CT', 'extentz', [-5 +8], 'guess', [0,0,0,0,0],...
    %       'figure',100,'figure_filename',fname));
    %     res = cell(1,4);
    %     e1 = htransform_vectors(inv(A2CT), rot_f4r);
    %     e2 = htransform_vectors(inv(A2CT), rot_f4r1);
    %     for ii=1:4
    %       res{ii}.ends(1,:) = e1(ii,:);
    %       res{ii}.ends(2,:) = e2(ii,:);
    %     end
    %
    %     new_image = create_new_image('','XYZ',[]);
    %     for ii=1:length(res)
    %       new_image.data = res{ii}.ends;
    %       new_image.Name = sprintf('CTFID%i',ii);
    %       arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    %     end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Register MRI Image
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.RegisterMRI.checkbox.Value
    AddProcessingOptions(handles, 'RegisterMRI');
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterMRI');
    handles.EPRf4Pars.figure_filename = fname;
    handles.MRIf4Pars.figure_filename = fname;


    % refit MRI fiducials
    output_list = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listMRI = arbuz_FindImage(handles.hGUI, output_list, 'InName', 'FID', {'data','Anative','A'});
    scale = hmatrix_scale([0.98, 0.98, 1]);

    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listEPR = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});
    
    CheckData(output_listEPR, sprintf('EPR image epr-fid-auto was not found !'));
    CheckData(output_listMRI, sprintf('MRI image FID was not found !'));
    
    disp('Refitting MRI');
    [AMR, A2MR, rotMR_f4r, rotMR_f4r1] = f4model(output_listMRI{1}.data, output_listMRI{1}.Anative, ...
      handles.MRIf4Pars);

    disp('Refitting EPR');
    [AEPR, A2EPR, rotEPR_f4r, rotEPR_f4r1] = f4model(output_listEPR{1}.data, output_listEPR{1}.Anative, ...
      handles.EPRf4Pars);

    arbuz_SetImage(handles.hGUI, image_MRIax, 'A', AMR \ AEPR);
    arbuz_SetImage(handles.hGUI, image_MRIax, 'Aprime', eye(4));
    Status('IMRTRegMRIax') = true;
    arbuz_set(handles.hGUI, 'Status', Status);

    if 1==0
      figure(1000); clf; hold on
      for ii=1:4
        P1 = htransform_vectors(inv(AMR), rotMR_f4r);
        P2 = htransform_vectors(inv(AMR), rotMR_f4r1);
        plot3([P1(ii,1), P2(ii,1)],...
          [P1(ii,2), P2(ii,2)],...
          [P1(ii,3), P2(ii,3)]); hold on
      end
      for ii=1:4
        P1 = htransform_vectors(inv(AEPR), rotEPR_f4r);
        P2 = htransform_vectors(inv(AEPR), rotEPR_f4r1);
        plot3([P1(ii,1), P2(ii,1)],...
          [P1(ii,2), P2(ii,2)],...
          [P1(ii,3), P2(ii,3)], '.'); hold on
      end
      legend({'MRI1','MRI2','MRI3','MRI4','EPR1','EPR2','EPR3','EPR4'})
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Register CT Image
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.RegisterCT.checkbox.Value
    AddProcessingOptions(handles, 'RegisterCT');
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterCT');
    handles.EPRf4Pars.figure_filename = fname;
    handles.CTf4Pars.figure_filename = fname;

    output_list = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listCT = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'ct-fid-auto', {'data','Anative'});

    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listEPR = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});

    CheckData(output_listEPR, sprintf('EPR image epr-fid-auto was not found !'));
    CheckData(output_listCT, sprintf('CT image ct-fid-auto was not found !'));

    disp('Refitting CT');
    [ACT, A2CT] = f4model(output_listCT{1}.data, output_listCT{1}.Anative, ...
      handles.CTf4Pars);
    
    disp('Refitting EPR');
    [AEPR, A2EPR] = f4model(output_listEPR{1}.data, output_listEPR{1}.Anative, ...
      handles.EPRf4Pars);

    arbuz_SetImage(handles.hGUI, image_CT, 'A', ACT \ AEPR);
    arbuz_SetImage(handles.hGUI, image_CT, 'Aprime', eye(4));

    handles.PrepData.Treatment_inCT = [];
    handles.PrepData.PO2_inCT = [];
    guidata(hObject, handles);

    Status('IMRTRegCT') = true;
    arbuz_set(handles.hGUI, 'Status', Status);
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Register PETDCE Image
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.RegisterPETDCE.checkbox.Value
    % registration matrix
    A = [ 0.0244 0.9970 0.1250 0; ...
      0.0539 -0.1261 0.9947 0;...
      0.9998 -0.0171 -0.0565 0;...
      2.2858 0.5260 -0.5568 1.0000];
    arbuz_SetTransformation(handles.hGUI, 'T-PET', 'PET', A);

    % registration matrix
    A = [0.0000   -0.0958    0.9954  0; ...
      0.0523   -0.9940   -0.0957     0; ...
      -0.9986   -0.0521   -0.0050    0; ...
      4.0522   -2.7304   -0.7143    1.0];

    arbuz_SetTransformation(handles.hGUI, 'T-PET', 'DCE', A);
    
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Create helper images for visualization
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.Visualization.checkbox.Value
    AddProcessingOptions(handles, 'Visualization');
    % MRI, Outline and fiducials
    if ~isempty(image_MRIax)
      output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-outline-autoSRF', {'SelectedColor','NotSelectedColor'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-outline-auto', {'data','ASLAVE'});
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist','Anative'});
        output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-outline-autoSRF', {'SelectedColor','NotSelectedColor'});
      end
      color = output_list2{1}.NotSelectedColor;
      color.EdgeColor = 'none';
      color.FaceAlpha = 0.3;
      color.FaceColor = [0, 0, 1];
      arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);

      output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-fid-autoSRF', {'SelectedColor','NotSelectedColor'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-fid-auto', {'data','ASLAVE'});
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist','Anative'});
        output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-fid-autoSRF', {'SelectedColor','NotSelectedColor'});
      end
      color = output_list2{1}.NotSelectedColor;
      color.FaceColor = [0, 0, 1];
      color.FaceAlpha = 0.3;
      color.EdgeColor = 'none';
      arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end

    % EPR, Outline and fiducials
    if ~isempty(image_PO2)
      output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'epr-outline-autoSRF', {'SelectedColor','NotSelectedColor'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'epr-outline-auto', {'data','ASLAVE'});
        if isempty(output_list2)
          output_list3 = arbuz_FindImage(handles.hGUI, image_PO2{1}, '', '', {'data'});
          new_image = create_new_image('epr-outline-auto','3DMASK',output_list3{1}.data > -99);
          CC = bwconncomp(new_image.data);
          items = cellfun(@(x) numel(x), CC.PixelIdxList);
          new_image.data = false(size(new_image.data));
          [~,maxidx] = max(items);
          new_image.data(CC.PixelIdxList{maxidx}) = true;
          arbuz_AddImage(handles.hGUI, new_image, image_PO2{1}.Image);
          image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'slavelist','Anative'});
          output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'epr-outline-auto', {'data','ASLAVE'});
        end
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'slavelist','Anative'});
        output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'epr-outline-autoSRF', {'SelectedColor','NotSelectedColor'});
      end
      color = output_list2{1}.NotSelectedColor;
      color.FaceColor = [1, 0, 1];
      color.EdgeColor = 'none';
      color.FaceAlpha = 0.3;
      arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end

    if ~isempty(image_FID)
      output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'Name', 'epr-fid-autoSRF', {'SelectedColor','NotSelectedColor'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'Name', 'epr-fid-auto', {'data','ASLAVE'});
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist','Anative'});
        output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'Name', 'epr-fid-autoSRF', {'SelectedColor','NotSelectedColor'});
      end
      color = output_list2{1}.NotSelectedColor;
      color.EdgeColor = 'none';
      color.FaceColor = [1, 0, 1];
      color.FaceAlpha = 0.3;
      arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end

    % Checking PET
    if ~isempty(image_PET)
      output_list2 = arbuz_FindImage(handles.hGUI, image_PET{1}.SlaveList, 'Name', 'pet-outline-auto', {'data','ASLAVE'});
      if ~isempty(output_list2)
        % create surface image
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_PET = arbuz_FindImage(handles.hGUI, 'master', 'Name', 'PET', {'slavelist'});
      end
      output_list2 = arbuz_FindImage(handles.hGUI, image_PET{1}.SlaveList, 'Name', 'pet-outline-autoSRF', {'NotSelectedColor','ASLAVE'});
      color = output_list2{1}.NotSelectedColor;
      color.EdgeColor = 'none';
      color.FaceColor = [0, 0, 0];
      color.FaceAlpha = 0.3;
      arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end

    % CT fiducials
    CTFIDAUTO = 'ct-fid-auto'; CTFIDAUTOSRF = [CTFIDAUTO,'SRF'];
    if ~isempty(image_CT)
      output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', CTFIDAUTOSRF, {'SelectedColor','NotSelectedColor'});
      if isempty(output_list2)
        % create surface image
        disp('Creating new CT FID surface');
        output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', CTFIDAUTO, {'data','ASLAVE'});
        if ~isempty(output_list2)
          [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
          new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
          new_image.A = output_list2{1}.Aslave;
          arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
          image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'slavelist','Anative'});
          output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'InName', CTFIDAUTOSRF, {'SelectedColor','NotSelectedColor'});
        end
      end
      if ~isempty(output_list2)
        color = output_list2{1}.NotSelectedColor;
        color.EdgeColor = 'none';
        color.FaceColor = [0, 1, 0];
        color.FaceAlpha = 0.3;
        arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
        arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
        arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
      end
    end

    CTOUTLINEAUTO = 'ct-outline-auto'; CTOUTLINEAUTOSRF = [CTOUTLINEAUTO,'SRF'];
    if ~isempty(image_CT)
      output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', CTOUTLINEAUTOSRF, {'SelectedColor','NotSelectedColor'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', CTOUTLINEAUTO, {'data','ASLAVE'});
        if ~isempty(output_list2)
          [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
          new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
          new_image.A = output_list2{1}.Aslave;
          arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
          image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'slavelist','Anative'});
          output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', CTOUTLINEAUTOSRF, {'SelectedColor','NotSelectedColor'});
        end
      end
      if ~isempty(output_list2)
        color = output_list2{1}.NotSelectedColor;
        color.EdgeColor = 'none';
        color.FaceColor = 'green';
        color.FaceAlpha = 0.3;
        arbuz_SetImage(handles.hGUI, output_list2, 'SelectedColor', color);
        arbuz_SetImage(handles.hGUI, output_list2, 'NotSelectedColor', color);
        arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
      end
    end
    Status('IMRTVisual') = true;
    arbuz_set(handles.hGUI, 'Status', Status);
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Statistics on the current experiment
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.GeneralStatistics.checkbox.Value
    AddProcessingOptions(handles, 'GeneralStatistics');
    events = {};
    for ii=1:length(handles.Files)
      filename = handles.(handles.Files{ii}).edit.String;
      if exist(filename, 'file')
        events{end+1}.ev = handles.Files{ii};
        file = dir(filename);
        events{end}.time =  datetime(file.date,'InputFormat','dd-MMM-yyyy HH:mm:ss');
        events{end}.offset = 1;
      end
    end
    files = dir([handles.eProjectPath.String, filesep, '*.mat']);
    for ii=1:length(files)
      events{end+1}.ev = files(ii).name;
      events{end}.time =  datetime(files(ii).date,'InputFormat','dd-MMM-yyyy HH:mm:ss');
      events{end}.offset = 3.5;
    end
    files = dir([handles.eProjectPath.String, filesep, '*.stl']);
    for ii=1:length(files)
      events{end+1}.ev = files(ii).name;
      events{end}.time =  datetime(files(ii).date,'InputFormat','dd-MMM-yyyy HH:mm:ss');
      events{end}.offset = 6;
    end
    files = dir([handles.eProjectPath.String, filesep, '*.ini']);
    for ii=1:length(files)
      events{end+1}.ev = files(ii).name;
      events{end}.time =  datetime(files(ii).date,'InputFormat','dd-MMM-yyyy HH:mm:ss');
      events{end}.offset = 6;
    end

    imfig = figure; hold on
    for ii=1:length(events)
      dt = events{ii}.time - events{1}.time;
      if hours(dt) < 12
        plot(hours(dt)*[1,1], -[0.6,0.1]+events{ii}.offset, 'b');
        text(hours(dt), events{ii}.offset, events{ii}.ev,'Rotation',90,'interpreter','none')
      end
    end
    axis([-Inf,Inf,0,12]);
    xlabel('Time [hours]');
    ylabel('Events');
    title('Time course of the experiment');
    fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
    savefig(imfig , fullfile(fpath, ['GeneralStatistics','1.fig']) , 'compact' )
    delete(imfig)
  end % end of GeneralStatistics

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % generate extended mask on EPR image
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %   if handles.PrepareTumorMask.checkbox.Value
  %     output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {});
  %     output_list2 = arbuz_FindImage(handles.hGUI, output_list2, 'Name', 'Tumor', {});
  %     if ~isempty(output_list2)
  %       % transfer MRI mask
  %       output_tumor = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'Tumor', {});
  %       if isempty(output_tumor)
  %         new_image = create_new_image('Tumor','3DMASK',[]);
  %         arbuz_AddImage(handles.hGUI, new_image, image_PO2{1}.Image);
  %         image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'slavelist'});
  %         output_tumor = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'Tumor', {});
  %       end
  %       res = arbuz_util_transform(handles.hGUI, output_list2, output_tumor{1}, []);
  %       arbuz_AddImage(handles.hGUI, res, image_PO2{1}.Image);
  %
  %       % fake out image for a now
  %       res.Name = 'Tumor_out';
  %       arbuz_AddImage(handles.hGUI, res, image_PO2{1}.Image);
  %     end
  %   end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Prepare PET data
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.PreparePETData.checkbox.Value
    AddProcessingOptions(handles, 'PreparePETData');

    tmp_time = safeget(handles.PreparePETData.options, 'PreparationTime', '');
    parameters.PreparationTime = datetime(tmp_time); handles.PreparePETData.options.PreparationTime = datestr(parameters.PreparationTime);
    tmp_time = safeget(handles.PreparePETData.options, 'InjectionTime', '');
    parameters.InjectionTime = datetime(tmp_time); handles.PreparePETData.options.InjectionTime = datestr(parameters.InjectionTime);
    tmp_time = safeget(handles.PreparePETData.options, 'PostInjectionTime', '');
    parameters.PostInjectionTime = datetime(tmp_time); handles.PreparePETData.options.PostInjectionTime = datestr(parameters.PostInjectionTime);
    tmp_time = safeget(handles.PreparePETData.options, 'AcquisitionStartTime', '');
    parameters.AcquisitionStartTime = datetime(tmp_time); handles.PreparePETData.options.AcquisitionStartTime = datestr(parameters.AcquisitionStartTime);
    tmp_time = safeget(handles.PreparePETData.options, 'AcquisitionEndTime', '');
    parameters.AcquisitionEndTime = datetime(tmp_time); handles.PreparePETData.options.AcquisitionEndTime = datestr(parameters.AcquisitionEndTime);
    Status('PreparePETData') = handles.PreparePETData.options;
    arbuz_set(handles.hGUI, 'Status', Status);

    parameters.SyringeDose = safeget(handles.PreparePETData.options, 'SyringeDose', 0);
    parameters.LeftoverDose = safeget(handles.PreparePETData.options, 'LeftoverDose', 0);
    parameters.MouseWeight = safeget(handles.PreparePETData.options, 'MouseWeight', 0);

    image_PET = arbuz_FindImage(handles.hGUI, image_PET, '', '', {'AShow','slavelist','data'});
    SUV = imrt_PET2SUV(image_PET{1}.data, parameters);
    arbuz_AddImage(handles.hGUI, create_new_image('SUV','BIN',SUV), image_PET{1}.Image);
    arbuz_UpdateInterface(handles.hGUI);

    PETpO2 = imrt_SUV2PO2(SUV, parameters);
    arbuz_AddImage(handles.hGUI, create_new_image('pO2','BIN',PETpO2), image_PET{1}.Image);
    arbuz_UpdateInterface(handles.hGUI);

  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Prepare image data
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.PrepareImageData.checkbox.Value
    AddProcessingOptions(handles, 'PrepareImageData');
    opts = handles.PrepareImageData.options;

    image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'AShow','slavelist'});

    % Need to set
    % SourceImage - image containing 
    % Source_pO2 - Image containing pO2
    % Source_Hypoxia - Image contaioning hypoxia 
    % Source_Tumor   - Image containing Tumor
    switch handles.LoadedParameters.Protocol
      case 2
        Source_pO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'data','slavelist'});
        image_PO2_tumor = arbuz_FindImage(handles.hGUI, Source_pO2{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
        if isempty(image_PO2_tumor) || safeget(opts, 'use_mri_mask', 1)
          fprintf('Tumor mask will be transfered from MRI.\n'); 
          MRI_Tumor = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
          CheckData(MRI_Tumor, 'Tumor mask was not found at MRI.');
          res = arbuz_util_transform(handles.hGUI, MRI_Tumor{1}, Source_pO2{1}, []);
          arbuz_AddImage(handles.hGUI, create_new_image('Tumor','3DMASK', res.data), image_PO2{1}.Image);
          arbuz_UpdateInterface(handles.hGUI);

          % reload
          Source_pO2  = arbuz_FindImage(handles.hGUI, image_PO2, 'ImageType', 'PO2_pEPRI', {'data','slavelist'});
          Source_Tumor = arbuz_FindImage(handles.hGUI, Source_pO2{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
        end

        Data_pO2 = Source_pO2{1}.data;
        Data_Tumor = Source_Tumor{1}.data;
        Source_Treatment = Source_pO2;
        Data_Treatment = Data_pO2 <= 10 & Data_pO2 > -25;

        Source_Tumor_Out = arbuz_FindImage(handles.hGUI, Source_pO2{1}.SlaveList, 'Name', safeget(opts, 'use_mask', 'Tumor'), {'data', 'AShow'});
        if isempty(Source_Tumor_Out), error('Tumor mask was not found.'); end
        Data_Tumor_Out = Source_Tumor_Out{1}.data;

        % name_of_tumor_mask = handles.PrepareImageData.options.use_mask;
        % image_PO2_tumor_out = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', name_of_tumor_mask, {'data', 'AShow'});
        % if isempty(image_PO2_tumor_out)
        %   if strcmp(name_of_tumor_mask, 'Tumor_OER')
        %     % generate OER style addition
        %     se = strel('sphere',1);
        %     T = image_PO2_tumor{1}.data;
        %     H = image_PO2{1}.data <= 10 & image_PO2{1}.data > -25;
        %     DRH = (imdilate(T, se) & ~T) & H; % hypoxic rim
        %     TRH = imdilate(T & H, se) & ~T; % hypoxia underlying rim
        %     Tumor_OER = T | (DRH & TRH);
        % 
        %     new_image = create_new_image('','3DMASK',[]);
        %     new_image.data = Tumor_OER;
        %     new_image.Name = name_of_tumor_mask;
        %     arbuz_AddImage(handles.hGUI, new_image, image_PO2{1}.Image);
        %     % Re-read pO2 and masks
        %     image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'data','slavelist'});
        %     image_PO2_tumor_out = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', name_of_tumor_mask, {'data', 'AShow'});
        %   else
        %     error(sprintf('pO2 image: %s mask was not found.', name_of_tumor_mask));
        %   end
        % end
      case 3 % treatment baset on PET pO2
        Hypoxia_Origin  = arbuz_FindImage(handles.hGUI, 'master', 'Name', 'PET', {'data','slavelist'});
        SlaveList = Hypoxia_Origin{1}.SlaveList;
        Source_pO2 = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'pO2', {'data','slavelist'});
        if isempty(Source_pO2), error('PET pO2 was not found.'); end
        Data_pO2 = Source_pO2{1}.data;
        Data_Treatment = Source_pO2{1}.data <= 10 & Source_pO2{1}.data > -25;

        Source_Tumor = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
        if isempty(Source_Tumor) || safeget(opts, 'use_mri_mask', 1)
          fprintf('Tumor mask will be transfered from MRI.\n'); 
          MRI_Tumor = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
          CheckData(MRI_Tumor, 'Tumor mask was not found at MRI.');
          res = arbuz_util_transform(handles.hGUI, MRI_Tumor{1}, Hypoxia_Origin{1}, []);
          arbuz_AddImage(handles.hGUI, create_new_image('Tumor','3DMASK', res.data), Hypoxia_Origin{1}.Image);
          arbuz_UpdateInterface(handles.hGUI);

          % reload
          Hypoxia_Origin  = arbuz_FindImage(handles.hGUI, 'master', 'Name', 'PET', {'data','slavelist'});
          SlaveList = Hypoxia_Origin{1}.SlaveList;
          Source_Tumor = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
        end
        Data_Tumor = Source_Tumor{1}.data;

        Source_Tumor_Out = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', safeget(opts, 'use_mask', 'Tumor'), {'data', 'AShow'});
        if isempty(Source_Tumor_Out), error('Tumor mask was not found.'); end
        Data_Tumor_Out = Source_Tumor_Out{1}.data;
      case 4 % treatment based on SUV and threshold
        Hypoxia_Origin  = arbuz_FindImage(handles.hGUI, 'master', 'Name', 'PET', {'data','slavelist'});
        SlaveList = Hypoxia_Origin{1}.SlaveList;
        Source_pO2 = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'pO2', {'data','slavelist'});
        Data_pO2 = Source_pO2{1}.data;

        Source_Treatment = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'SUV', {'data','slavelist'});
        if isempty(Source_Treatment), error('PET SUV was not found.'); end
                
        Source_Tumor = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
        if isempty(Source_Tumor) || safeget(opts, 'use_mri_mask', 1)
          fprintf('Tumor mask will be transfered from MRI.\n'); 
          MRI_Tumor = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
          CheckData(MRI_Tumor, 'Tumor mask was not found at MRI.');
          res = arbuz_util_transform(handles.hGUI, MRI_Tumor{1}, Hypoxia_Origin{1}, []);
          arbuz_AddImage(handles.hGUI, create_new_image('Tumor','3DMASK', res.data), Hypoxia_Origin{1}.Image);
          arbuz_UpdateInterface(handles.hGUI);

          % reload
          Hypoxia_Origin  = arbuz_FindImage(handles.hGUI, 'master', 'Name', 'PET', {'data','slavelist'});
          SlaveList = Hypoxia_Origin{1}.SlaveList;
          Source_Tumor = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
        end
        Data_Tumor = Source_Tumor{1}.data;

        Source_Tumor_Out = arbuz_FindImage(handles.hGUI, SlaveList, 'Name', safeget(opts, 'use_mask', 'Tumor'), {'data', 'AShow'});
        if isempty(Source_Tumor_Out), error('Tumor mask was not found.'); end
        Data_Tumor_Out = Source_Tumor_Out{1}.data;

        mean_tumor_SUV = mean(Source_Treatment{1}.data(Data_Tumor));
        treatment_threshold = safeget(opts, 'threshold', 1.4);
        Data_Treatment = Source_Treatment{1}.data >= treatment_threshold*mean_tumor_SUV;


    end
    vis_treatment = Data_Treatment;
    vis_treatment(~Data_Tumor)=false;
    arbuz_AddImage(handles.hGUI, create_new_image('Treatment','3DMASK', 2*vis_treatment+Data_Tumor), Hypoxia_Origin{1}.Image);
    arbuz_UpdateInterface(handles.hGUI);

    res = arbuz_util_transform_data(handles.hGUI, Source_pO2{1}, Data_pO2, false, image_CT{1}, []);
    handles.PrepData.PO2_inCT = res.data;
    res = arbuz_util_transform_data(handles.hGUI, Source_Tumor{1}, Data_Tumor, true, image_CT{1}, []);
    handles.PrepData.Tumor_inCT = res.data;
    res = arbuz_util_transform_data(handles.hGUI, Source_Tumor_Out{1}, Data_Tumor_Out, true, image_CT{1}, []);
    handles.PrepData.TumorExt_inCT = res.data;
    res = arbuz_util_transform_data(handles.hGUI, Source_Treatment, Data_Treatment, true, image_CT{1}, []);
    handles.PrepData.Treatment_inCT = res.data & handles.PrepData.TumorExt_inCT;

    % CT  = arbuz_FindImage(handles.hGUI, image_CT{1}, '', '', {'data','slavelist'});
    % handles.PrepData.CT = CT{1}.data;
    % ibGUI(handles.PrepData)

    %    ibGUI((handles.PO2_inCT ~= -100) + handles.Tumor_inCT + handles.PrepData.Treatment_inCT)
    %    ibGUI((handles.PO2_inCT <= 10 & handles.PO2_inCT >= 0) + handles.Tumor_inCT + handles.PrepData.Treatment_inCT)

    handles.project_state = arbuz_get(handles.hGUI, 'state');
    guidata(hObject, handles);

    fprintf('PrepData are ready.\n');

    % export_data = safeget(opts, 'export_data',0);
    % if export_data
    %   [filename, pathname] = uiputfile({'*.mat'}, 'Save as');
    %   if ~isempty(filename)
    %     res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-cast-auto', {'data'});
    %     material_mask = res{1}.data*3;
    %     res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-outline-auto', {'data'});
    %     material_mask = material_mask + res{1}.data;
    %     res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-mousebed-auto', {'data'});
    %     material_mask = material_mask + res{1}.data*2;
    %     res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-fid-auto', {'data'});
    %     material_mask = material_mask + res{1}.data*3;
    % 
    % 
    %     %  Tumor
    %     exp.Tumor = handles.PrepData.TumorExt_inCT;
    %     %  Hypoxia
    %     exp.Hypoxia = handles.PrepData.Treatment_inCT;
    %     % Leg segmentation
    %     exp.LegCTSegmentation = material_mask;
    % 
    %     % Beam Center
    %     exp.beam_center = epr_maskcm(handles.PrepData.Treatment_inCT);
    %     %         exp.beam_center = exp.beam_center([2,1,3]);
    % 
    %     save(fullfile(pathname, filename), '-struct', 'exp');
    %   end
    % end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Whole field dose planning
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.WholeFieldPlanning.checkbox.Value
    AddProcessingOptions(handles, 'WholeFieldPlanning');

    image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'data', 'AShow','slavelist'});
    opts = handles.WholeFieldPlanning.options;
    prescription  = safeget(opts, 'prescription_Gy', 50);
    Field_size_mm = safeget(opts, 'Field_size_mm',35);
    bone_segmentation_retrofix = safeget(opts, 'bone_segmentation_retrofix',0);
    opts = handles.SegmentCTfiducials.options;
    noise_density_max = safeget(opts, 'noise_density_max', 450);

    Gantry_angles = [90 -90];
    d = round((Field_size_mm/0.025)/1.32); %Standard Whole field radius in Bev sized pixels.
    %Divide by the mag factor because the planning assumes you are talking about exit plane
    Bev_masks=cell(1,length(Gantry_angles));
    for ii =1:length(Gantry_angles)
      [ Bev_masks{ii}.Dilated_boost_map ] = epr_create_circular_mask( [d*2 d*2] , d );
    end

    res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-cast-auto', {'data'});
    material_mask = res{1}.data*3;
    res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-outline-auto', {'data'});
    material_mask = material_mask + res{1}.data;
    res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-mousebed-auto', {'data'});
    material_mask = material_mask + res{1}.data*2;
    res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-fid-auto', {'data'});
    material_mask = material_mask + res{1}.data*3;
    % remove overlap
    material_mask(material_mask > 3.1) = 1;
    material_mask(image_CT{1}.data > noise_density_max & material_mask == 0) = 1;

    if bone_segmentation_retrofix
      res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-bone-auto', {'data'});
      material_mask(res{1}.data > 0.5) = 0;
      res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-fid-auto', {'data'});
      material_mask(res{1}.data > 0.5) = 1;
    end
    %     ibGUI(material_mask)

    Center_postion = 1;
    [~, ~, beamtimes, imfig] = ...
      maskbev_depths_func_general(image_CT{1}.data, zeros(size(material_mask)), material_mask, Bev_masks, Gantry_angles, prescription, Center_postion );

    fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
    fparts = strsplit(handles.eProjectPath.String, filesep);
    Experiment_name = fparts{end};
    AddMessage(handles, '#WholeFieldAngle',sprintf('%5.1f  ', Gantry_angles));
    fname = sprintf('%s',Experiment_name,'_Whole_field');
    AddMessage(handles, '#WholeFieldBeamTime',[sprintf('prescription=%5.2fGy beamtimes=', prescription),sprintf('%5.2f  ', beamtimes)], true);

    Beam_plan_INI_write_V3(fpath, fname, Gantry_angles, prescription/length(Gantry_angles), beamtimes);
    fprintf('Project %s: whole field ini file is created.\n',Experiment_name);

    savefig(imfig(1) , fullfile(fpath, ['WholeFieldPlanning','1.fig']) , 'compact' )
    savefig(imfig(2), fullfile(fpath, ['WholeFieldPlanning','2.fig']) , 'compact' )
    delete(imfig(1))
    delete(imfig(2))
  end


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Treatment dose planning
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.BoostFieldDosePlanning.checkbox.Value
    AddProcessingOptions(handles, 'BoostFieldDosePlanning');

    if isempty(handles.PrepData.Treatment_inCT) || isempty(handles.PrepData.PO2_inCT) || project_state ~= handles.project_state
      % request reprocesssing
      handles.PrepareImageData.checkbox.Value = true;
      pbRunAll_Callback(hObject, [], handles);
      handles = guidata(handles.figure1);
    else

      opts = handles.BoostFieldDosePlanning.options;
      bone_segmentation_retrofix = safeget(opts, 'bone_segmentation_retrofix',0);

      image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'data', 'AShow','slavelist'});

      Gantry_angles = 90 : -72 :-270+72;
      beam_center = epr_maskcm(handles.PrepData.Treatment_inCT);
      beam_center = beam_center([2,1,3]);

      Bev_masks = cell(length(Gantry_angles), 1);
      for ii=1:length(Gantry_angles)
        Bev_masks{ii}.Angle = Gantry_angles(ii);
        [Bev_masks{ii}.Hypoxia, Bev_masks{ii}.Boost_bev_volume] = ...
          imrt_maskbev( Gantry_angles(ii), handles.PrepData.Treatment_inCT, beam_center, []);
        [Bev_masks{ii}.Tumor] = ...
          imrt_maskbev( Gantry_angles(ii), handles.PrepData.Tumor_inCT, beam_center, [] );
      end
      Bev_size = cellfun(@(x) x.Boost_bev_volume, Bev_masks);
      [~,min_idx] = min(Bev_size);
      Port_Gantry_Angle = Gantry_angles(min_idx);

      imfig = imrt_show_bev(Bev_masks, min_idx);
      fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
      savefig(imfig , fullfile(fpath, ['BoostFieldDosePlanning','1.fig']) , 'compact' )
      delete(imfig);

      Gantry_angles = [Port_Gantry_Angle, Port_Gantry_Angle - 180 + ((Port_Gantry_Angle < -90)*360)];

      fprintf('Generate Bevs for the target volume\n')
      pars = struct('Plane2PlugScale', 1.26, 'BoostMargin', opts.boost_margin);

      Bev_masks = cell(length(Gantry_angles), 1);
      for ii =  1:length(Gantry_angles)
        Bev_masks{ii}.Angle = Gantry_angles(ii);
        [Bev_masks{ii}.Boost_map, Bev_masks{ii}.Boost_bev_volume] = ...
          imrt_maskbev( Gantry_angles(ii), handles.PrepData.Treatment_inCT, beam_center, []);
        Bev_masks{ii}.Dilated_boost_map = imrt_tranform_mask(Bev_masks{ii}.Boost_map,'boost',pars);
      end
      prescription = safeget(opts,'prescription_Gy',11);

      res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-cast-auto', {'data'});
      material_mask = res{1}.data*3;
      res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-outline-auto', {'data'});
      material_mask = material_mask + res{1}.data;
      res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-mousebed-auto', {'data'});
      material_mask = material_mask + res{1}.data*2;
      res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-fid-auto', {'data'});
      material_mask = material_mask + res{1}.data*3;
      % remove overlap
      material_mask(material_mask > 3.1) = 1;
      material_mask(image_CT{1}.data > opts.noise_density_max & material_mask == 0) = 1;

      if bone_segmentation_retrofix
        res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-bone-auto', {'data'});
        if ~isempty(res), material_mask(res{1}.data > 0.5) = 0; end
        res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-fid-auto', {'data'});
        if ~isempty(res), material_mask(res{1}.data > 0.5) = 1; end
      end

      Center_postion = 1;
      [~, ~, beamtimes, imfig] = ...
        maskbev_depths(image_CT{1}.data, handles.PrepData.Treatment_inCT, material_mask, Bev_masks, Gantry_angles, prescription, Center_postion );

      fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
      fparts = strsplit(handles.eProjectPath.String, filesep);
      Experiment_name = fparts{end};

      switch  opts.boost_1_aboost_2
        case {1,3}, fname = sprintf('%s',Experiment_name,'_Boost');
        case 2, beamtimes = beamtimes * (12/7); % anti-boost
          fname = sprintf('%s',Experiment_name,'_AntiBoost');
      end

      %     Target = epr_maskcm(handles.PrepData.Treatment_inCT);
      %     opts.A = image_CT{1}.Ashow;
      %     [~, ~, beamtimes, imfig] = ...
      %       depth_dose_calculation(image_CT{1}.data, material_mask, Target, Bev_masks,opts);

      % output an INI file to the outpath for loading into the pilot software.
      Beam_plan_INI_write_V3(fpath, fname, Gantry_angles, prescription/length(Gantry_angles), beamtimes);
      fprintf('Project %s: whole field ini file is created.\n',Experiment_name);
      AddMessage(handles, '#BoostFieldAngle',sprintf('%5.1f  ', Gantry_angles));
      AddMessage(handles, '#BoostFieldBeamTime',[sprintf('prescription=%5.1fGy beamtimes=',prescription),sprintf('%5.2f  ', beamtimes)], true);

      savefig(imfig(1) , fullfile(fpath, ['BoostFieldDosePlanning','2.fig']) , 'compact' )
      savefig(imfig(2) , fullfile(fpath, ['BoostFieldDosePlanning','3.fig']) , 'compact' )
      delete(imfig(1))
      delete(imfig(2))
    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Treatment planning
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.PlanTreatment.checkbox.Value

    if isempty(handles.PrepData.Treatment_inCT) || isempty(handles.PrepData.PO2_inCT) || project_state ~= handles.project_state
      % request reprocesssing
      handles.PrepareImageData.checkbox.Value = true;
      pbRunAll_Callback(hObject, [], handles);
      handles = guidata(handles.figure1);
    else
      AddProcessingOptions(handles, 'PlanTreatment');

      WL_shift = Winston_Lutz_corrections();

      Target = handles.PrepData.Treatment_inCT;
      beam_center = calculate_center(handles.PrepData.Treatment_inCT);
      beam_center = beam_center([2,1,3]);

      %     beam_center = size(Target)/2;

      opts = handles.PlanTreatment.options;

      PlugSize = safeget(opts, 'Plug_size', 16);
      strPlugSize = [num2str(PlugSize), 'mm'];

      Gantry_angles = 90 : -72 :-270+72;
      for ii=1:length(Gantry_angles)
        Bev_masks{ii}.Angle = Gantry_angles(ii);
        [Bev_masks{ii}.Hypoxia, Bev_masks{ii}.Boost_bev_volume, pix_beam_center] = ...
          imrt_maskbev( Gantry_angles(ii), Target, beam_center, []);
        [Bev_masks{ii}.Tumor] = ...
          imrt_maskbev( Gantry_angles(ii), handles.PrepData.TumorExt_inCT, beam_center, [] );
      end
      Bev_size = cellfun(@(x) x.Boost_bev_volume, Bev_masks);
      [~,min_idx] = min(Bev_size);
      Port_Gantry_Angle = Gantry_angles(min_idx);

      imfig = imrt_show_bev(Bev_masks, min_idx, pix_beam_center,PlugSize);
      fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
      savefig(imfig , fullfile(fpath, ['PlanTreatment','1.fig']) , 'compact' )
      delete(imfig);

      Gantry_angles = [Port_Gantry_Angle, Port_Gantry_Angle - 180 + ((Port_Gantry_Angle < -90)*360)];

      disp('Generate Bevs for the target volume')

      Bev_masks = cell(length(Gantry_angles), 1);
      for ii=1:length(Gantry_angles)
        Bev_masks{ii}.Angle = Gantry_angles(ii);
        [Bev_masks{ii}.Hypoxia] = ...
          imrt_maskbev( Gantry_angles(ii) , Target, beam_center, []);
        [Bev_masks{ii}.Tumor] = ...
          imrt_maskbev( Gantry_angles(ii) , handles.PrepData.TumorExt_inCT, beam_center, [] );
      end

      %Writes the bed shift to a text file. Important to note that X= j and Y = i This is the classic problem with Matlab having colums be the first dim.
      fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
      fparts = strsplit(handles.eProjectPath.String, filesep);
      Experiment_name = fparts{end};
      fname = sprintf('%s',Experiment_name,'_Bed_shift.txt');

      Bed_shift_textfile = fullfile(fpath, fname);
      icen=beam_center(2); jcen=beam_center(1); kcen=beam_center(3);
      imid = (size(Target,2)/2); jmid = (size(Target,1)/2); kmid = (size(Target,3)/2);
      fid = fopen(Bed_shift_textfile, 'w+');
      str = sprintf(' Move the bed by  X  %6.5f  Y   %6.5f   Z    %6.5f', ...
        (jcen - jmid)/-100, (icen - imid)/100, (kcen - kmid)/100);
      fprintf(fid, str);
      fclose(fid);

      fprintf('setting Boost margin to %4.2f\n',opts.boost_margin);
      fprintf('setting AntiBoost margin to %4.2f\n',opts.antiboost_margin);

      pars = struct('Plane2PlugScale', 1.26, ...
        'BoostMargin', opts.boost_margin, ...
        'ABoostMargin', opts.antiboost_margin);
      for ii =  1:length(Bev_masks)
        Bev_masks{ii}.BoostMargin = opts.boost_margin;
        Bev_masks{ii}.Boost = imrt_tranform_mask(Bev_masks{ii}.Hypoxia,'boost',pars);
        Bev_masks{ii}.Boost_bev_volume = numel(find(Bev_masks{ii}.Boost));
      end

      Scale_factor = [0.025, 0.025, 1]; % CT image pixel in [mm]

      switch opts.boost_1_aboost_2
        case 1 % 'Boost'

          disp('Writing SCAD files and rendering plugs for Boost')
          for ii = 1:length(Bev_masks)
            filename = fullfile(fpath, sprintf('%s_Boost_%i_%ideg', Experiment_name, ii,Bev_masks{ii}.Angle));

            BEVTarget = Bev_masks{ii}.Boost;

            %Get the Winston-Lutz (UV shifts in the plug positon from a table)
            idx  = WL_shift.Gantry_angle == Bev_masks{ii}.Angle;
            UVShift = [WL_shift.Plug_X(idx),WL_shift.Plug_Y(idx)];

            imrt_openscad('beam', BEVTarget, Scale_factor, UVShift, strPlugSize, filename);
            AddMessage(handles, '#BoostScadFile',sprintf('%5.2f %s', Bev_masks{ii}.Angle, filename), true);
          end

        case 2 % 'AntiBoost'

          disp('Writing SCAD files and rendering plugs for AntiBoost')
          algorithm = safeget(opts, 'antiboost_algorithm', 0);
          for ii = 1:length(Bev_masks)
            filename = fullfile(fpath, sprintf('%s_Anti_Boost_%i_%ideg', Experiment_name, ii,Bev_masks{ii}.Angle));

            pars.beam_square = Bev_masks{ii}.Boost_bev_volume;

            switch algorithm
              case 1
                [pars.slTumor] = Bev_masks{ii}.Tumor;
                Bev_masks{ii}.Antiboost = imrt_tranform_mask(Bev_masks{ii}.Hypoxia,'antiboost-ver2',pars);
              case 2
                [pars.slTumor] = Bev_masks{ii}.Tumor;
                Bev_masks{ii}.Antiboost = imrt_tranform_mask(Bev_masks{ii}.Hypoxia,'antiboost-ver3',pars);
              otherwise
                Bev_masks{ii}.Antiboost = imrt_tranform_mask(Bev_masks{ii}.Hypoxia,'antiboost',pars);
            end

            Bev_masks{ii}.ABoostMargin = opts.antiboost_margin;
            BEVTarget = Bev_masks{ii}.Antiboost;

            %Get the Winston-Lutz (UV shifts in the plug positon from a table)
            idx  = WL_shift.Gantry_angle == Bev_masks{ii}.Angle;
            UVShift = [WL_shift.Plug_X(idx),WL_shift.Plug_Y(idx)];

            imrt_openscad('shell', BEVTarget, Scale_factor, UVShift, strPlugSize, filename);
          end


        case 3 % OEC experiment calibrated to oxygenated

          disp('Writing SCAD files and rendering plugs for OER calibrated oxygen delivery')
          algorithm = safeget(opts, 'antiboost_algorithm', 0);
          for ii = 1:length(Bev_masks)
            filename = fullfile(fpath, sprintf('%s_Anti_Boost_%i_%ideg', Experiment_name, ii,Bev_masks{ii}.Angle));

            BEVTarget = Bev_masks{ii}.Boost;
            %             pars.beam_square = Bev_masks{ii}.Boost_bev_volume;
            %
            %             [pars.slTumor] = Bev_masks{ii}.Tumor;
            %             Bev_masks{ii}.Antiboost = imrt_tranform_mask(Bev_masks{ii}.Hypoxia,'antiboost-max',pars);
            %
            %             Bev_masks{ii}.ABoostMargin = opts.antiboost_margin;
            %             Target = Bev_masks{ii}.Antiboost;
            %
            % Get the Winston-Lutz (UV shifts in the plug positon from a table)
            idx  = WL_shift.Gantry_angle == Bev_masks{ii}.Angle;
            UVShift = [WL_shift.Plug_X(idx),WL_shift.Plug_Y(idx)];

            imrt_openscad('inv-beam', BEVTarget, Scale_factor, UVShift, strPlugSize, filename);
          end


      end

      imfig = imrt_show_bev(Bev_masks, -1, [0,0], PlugSize);
      fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
      savefig(imfig , fullfile(fpath, ['PlanTreatment','2.fig']) , 'compact' );
      if opts.boost_1_aboost_2 == 1
        savefig(imfig , fullfile(fpath, ['PlanTreatment-B.fig']) , 'compact' );
      elseif opts.boost_1_aboost_2 == 2
        savefig(imfig , fullfile(fpath, ['PlanTreatment-A',num2str(algorithm),'.fig']) , 'compact' );
      end
      delete(imfig);

      save(fullfile(fpath, 'production_data'), 'Bev_masks')

    end
  end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Coverage statistics
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.CoverageMap.checkbox.Value
    AddProcessingOptions(handles, 'CoverageMap');

    Target = handles.PrepData.Treatment_inCT;
    Tumor  = handles.PrepData.TumorExt_inCT;
    opts = handles.PlanTreatment.options;

    fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
    s1 = load(fullfile(fpath, 'production_data'), 'Bev_masks');
    Bev_masks = s1.Bev_masks;

    % approximate calculation of the coverage map without attenuation
    coverage_map = zeros(size(Tumor));
    switch opts.boost_1_aboost_2
      case 1
        for ii = 1:length(Bev_masks)
          MM = mask_bev_quant(Tumor, Target, Bev_masks{ii}.Boost, Bev_masks{ii}.Angle);
          coverage_map = coverage_map + MM*opts.prescription_Gy / 2;
        end
      case 2
        for ii = 1:length(Bev_masks)
          MM = mask_bev_quant(Tumor, Target, Bev_masks{ii}.Antiboost, Bev_masks{ii}.Angle);
          coverage_map = coverage_map + MM*opts.prescription_Gy / 2;
        end
    end
    % ibGUI(coverage_map)
    hypoxia_hit = numel(find((coverage_map > opts.prescription_Gy*0.9) & (handles.PrepData.Treatment_inCT & handles.PrepData.Tumor_inCT)));
    normoxia_hit = numel(find((coverage_map > opts.prescription_Gy*0.9) & (handles.PrepData.TumorExt_inCT & ~handles.PrepData.Treatment_inCT)));
    hypoxia_overall = numel(find(handles.Tumor_inCT & handles.PrepData.Treatment_inCT));
    normoxia_overall = numel(find((handles.Tumor_inCT & ~handles.PrepData.Treatment_inCT)));
    AddMessage(handles, '#CoverageMap',sprintf('hypoxia: hit %i overall %i ', hypoxia_hit, hypoxia_overall), true);
    AddMessage(handles, '#CoverageMap',sprintf('normoxia: hit %i overall %i ', normoxia_hit, normoxia_overall), true);

    save(fullfile(fpath, 'coverage_map'), 'coverage_map');
  end
catch e
  fprintf(2,'%s\n',getReport(e));
end
% update interface
PrepareThePlan(handles);
arbuz_UpdateInterface(handles.hGUI);
handles.panProcessing.ForegroundColor = 'black';
handles.panProcessing.Title = 'Processing';
set(handles.panProcessing.Children, 'Enable', 'on');
set(handles.panFiles.Children, 'Enable', 'on');
set(handles.uipanel1.Children, 'Enable', 'on');

% log file
ReadLog(handles);

% --------------------------------------------------------------------
function PrepareThePlan(handles)

project_state = arbuz_get(handles.hGUI, 'state');

for ii=1:length(handles.Files)
   handles.(handles.Files{ii}).inproject = false;
   is_image = arbuz_FindImage(handles.hGUI, 'master', 'Name', handles.(handles.Files{ii}).id, {'Slavelist'});
   if isempty(handles.(handles.Files{ii}).idslave)
     if ~isempty(is_image), handles.(handles.Files{ii}).inproject = true; end
   elseif ~isempty(is_image)
     is_slaveimage = arbuz_FindImage(handles.hGUI, is_image{1}.SlaveList, 'Name', handles.(handles.Files{ii}).idslave, {});
     if ~isempty(is_slaveimage), handles.(handles.Files{ii}).inproject = true; end
   end
end

% Assign files directory
PathName = handles.eProjectPath.String;
if ~isempty(PathName)
  for ii=1:length(handles.Files)
    fname = ReadFileName(handles, handles.Files{ii});
    if isempty(fname)
      fname = FindFileName(PathName, handles.Files{ii});
      if ~isempty(fname)
        set(handles.(handles.Files{ii}).edit, 'string', fname,'value', 1);
      else
        set(handles.(handles.Files{ii}).edit, 'string', fname,'value', -1);
      end
      % handles.(handles.Files{ii}).inproject = false;
    else
      if contains(handles.Files{ii}, 'EPRpO2other')
        set(handles.(handles.Files{ii}).edit, 'string', fname, 'value', 1);
      else
        set(handles.(handles.Files{ii}).edit, 'string', fname);
      end
      % handles.(handles.Files{ii}).inproject = true;
    end
  end
end

% files to be loaded and color of loaded
for ii=1:length(handles.Files)
  fname = handles.(handles.Files{ii}).edit.String;
  handles.(handles.Files{ii}).checkbox.Value = 0;
  if iscell(fname), fname = fname{1}; end
  if handles.(handles.Files{ii}).inproject
    set(handles.(handles.Files{ii}).edit, 'background',handles.color.inproject);
  elseif exist(fname, 'file') ~= 2
    set(handles.(handles.Files{ii}).edit, 'background',handles.color.incorrect);
  else
    set(handles.(handles.Files{ii}).edit, 'background',handles.color.correct);
    handles.(handles.Files{ii}).checkbox.Value = handles.(handles.Files{ii}).need_to_load;
  end
end

% processing steps
for ii=1:length(handles.Processings)
  handles.(handles.Processings{ii}).checkbox.Value = false;
  handles.(handles.Processings{ii}).checkbox.BackgroundColor = handles.color.correct;
end

% find out sequence
seq = arbuz_get(handles.hGUI, 'SEQUENCES');
handles.CreateRegistrationSequence.checkbox.Value = isempty(seq);
if ~isempty(seq)
  handles.CreateRegistrationSequence.checkbox.BackgroundColor = handles.color.inproject;
end

image_CT = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'DICOM3D', {});
image_CT = arbuz_FindImage(handles.hGUI, image_CT, 'InName', 'CT', {'slavelist','A'});
image_MRIax = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'MRI', {});
image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, 'InName', '_ax', {'slavelist','A'});
image_MRIsag = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'MRI', {});
image_MRIsag = arbuz_FindImage(handles.hGUI, image_MRIsag, 'InName', '_sag', {'slavelist','A'});
image_PO2 = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', 'PO2_pEPRI', {});
image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, 'InName', '_2', {'slavelist','A'});
image_FID = arbuz_FindImage(handles.hGUI, 'master', 'ImageType', '3DEPRI', {});
image_FID = arbuz_FindImage(handles.hGUI, image_FID, 'InName', 'FID', {'slavelist'});
image_PET = arbuz_FindImage(handles.hGUI, 'master', 'InName', 'PET', {'slavelist'});

Status = arbuz_get(handles.hGUI, 'Status');

isMriaxRegistration = false;

% image segmentation MRI axial
if ~isempty(image_MRIax)
  if isempty(image_MRIax{1}.SlaveList)
    handles.SegmentMRIaxFiducials.checkbox.Value = 1;
    handles.SegmentMRIaxFiducials.checkbox.BackgroundColor = handles.color.correct;
  else
    output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'InName', '-auto', {'Color'});
    handles.SegmentMRIaxFiducials.checkbox.Value = isempty(output_list2);
    if ~isempty(output_list2)
      isMriaxRegistration = ~isequal(image_MRIax{1}.A, eye(4));
      handles.SegmentMRIaxFiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
  end

  output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {});
  output_list2 = arbuz_FindImage(handles.hGUI, output_list2, 'InName', 'Tumor', {});
  if ~isempty(output_list2)
    handles.SegmentTumor.checkbox.BackgroundColor = handles.color.inproject;
  end
end

% image segmentation MRI
if ~isempty(image_MRIsag)
  handles.SegmentMRIsagFiducials.checkbox.Value = isempty(image_MRIsag{1}.SlaveList);
  if ~isempty(image_MRIsag{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_MRIsag{1}.SlaveList, 'InName', '-auto', {'color'});
    handles.SegmentMRIsagFiducials.checkbox.Value = isempty(output_list2);
    if ~isempty(output_list2)
      handles.SegmentMRIsagFiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% image segmentation FID EPR
if ~isempty(image_FID)
  handles.SegmentEPRfiducials.checkbox.Value = isempty(image_FID{1}.SlaveList);
  if ~isempty(image_FID{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'InName', 'auto', {'color'});
    handles.SegmentEPRfiducials.checkbox.Value = isempty(output_list2);
    if ~isempty(output_list2)
      handles.SegmentEPRfiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% image segmentation CT
isCTRegistration = false;
if ~isempty(image_CT)
  handles.SegmentCTfiducials.checkbox.Value = isempty(image_CT{1}.SlaveList);
  if ~isempty(image_CT{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'InName', '-auto', {'color'});
    handles.SegmentCTfiducials.checkbox.Value = isempty(output_list2);
    isCTRegistration = ~isequal(image_CT{1}.A, eye(4));
    if ~isempty(output_list2)
      handles.SegmentCTfiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% image registration
handles.RegisterMRI.checkbox.Value = ~isMriaxRegistration;
if isMriaxRegistration
  handles.RegisterMRI.checkbox.BackgroundColor = handles.color.inproject;
end

% image registration
handles.RegisterCT.checkbox.Value = ~isCTRegistration;
if isCTRegistration
  handles.RegisterCT.checkbox.BackgroundColor = handles.color.inproject;
end

% visualization
if Status.isKey('IMRTVisual') && Status('IMRTVisual') == true
  handles.Visualization.checkbox.BackgroundColor = handles.color.inproject;
end

% data preparation
if  ~isempty(image_PO2)
  if isempty(image_PO2{1}.SlaveList)
    %     output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'Tumor', {});
    %     handles.PrepareTumorMask.checkbox.Value = ~isempty(output_list2);
  else
    output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'Tumor_out', {});
    %     handles.PrepareTumorMask.checkbox.Value = isempty(output_list2);
    if ~isempty(output_list2)
      handles.PrepareTumorMask.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% display if data processed or not
if ~isempty(handles.PrepData.Treatment_inCT) && ~isempty(handles.PO2_inCT) &&  handles.project_state == project_state
  handles.PrepareImageData.checkbox.BackgroundColor = handles.color.inproject;
end

ReadLog(handles);

% --------------------------------------------------------------------
function [str, fields] = read_log_file(filename)
fields.anesth_iso = [];
fields.temperature = [];
fields.bpm = [];
fields.Qvalue = [];
fields.signal = [];

fid = fopen(filename,'r');
str = {};
if fid~=-1 %if the file doesn't exist ignore the reading code
  while ~feof(fid)
    ss = fgets(fid); ss = strtrim(ss);
    if ~isempty(ss)
      str{end+1} = ss;
      a = regexp(ss, '(?<date>\d\d:\d\d:\d\d)\s+(?<field>#\w+)\s+(?<value>[\d.eE]+)', 'names');
      try
        if ~isempty(a) && isfield(a, 'field')
          for ii=1:length(a)
            b = a(ii);
            switch b.field
              case '#anesth_iso'
                fields.anesth_iso(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                fields.anesth_iso(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#temperature'
                fields.temperature(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                fields.temperature(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#bpm'
                fields.bpm(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                fields.bpm(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#Qvalue'
                fields.Qvalue(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                fields.Qvalue(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#signal'
                fields.signal(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                fields.signal(end).time  = sum(tt.*[1;1/60;1/3600]);
            end
          end
        end
      catch err
        disp(err);
      end
    end
  end
  fclose(fid);
end

% --------------------------------------------------------------------
function ReadLog(handles)
IMRTfolder = handles.eFolder.String;
[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
[str, handles.monitoring] = read_log_file(fname);

set(handles.eLogOld,'String',str);
set(handles.eLog,'String',{})
guidata(handles.figure1, handles);

% try
%     % not every version of Java has jhEdit.anchorToBottom property
%     jhEdit = findjobj(handles.eLog);
%     jhEdit.anchorToBottom;
%     jhEdit = findjobj(handles.eLogOld);
%     jhEdit.anchorToBottom;
% catch
% end

DrawTimeline(handles)

% --------------------------------------------------------------------
function AppendLog(handles)
IMRTfolder = handles.eFolder.String;
[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
epr_mkdir(fileparts(fname));
fid = fopen(fname,'a+');
str = handles.eLog.String;

if fid~=-1 %if the file doesn't exist ignore the reading code
  for ii=1:length(str)
    fprintf(fid, '%s\n', str{ii});
  end
  fclose(fid);
end

% --------------------------------------------------------------------
function AddProcessingOptions(handles, opts)
str = handles.eLog.String;
selopts = handles.(opts).options;
message = sprintf('%s #%s ', datestr(datetime, 'HH:MM:SS'), opts);
if ~isempty(selopts)
  fn = fieldnames(selopts);
  for ii=1:length(fn)
    message = strcat(message, sprintf(' %s %4.2f', fn{ii}, selopts.(fn{ii})));
  end
end
str{end+1} = message;
handles.eLog.String = str;
AppendLog(handles);
ReadLog(handles);

% --------------------------------------------------------------------
function pbSaveLog_Callback(hObject, ~, handles)
AppendLog(handles);
ReadLog(handles);

% --------------------------------------------------------------------
function AddMessage(handles, message_tag, message, is_save)
if ~exist('is_save', 'var'), is_save = false; end
str = handles.eLog.String;
str{end+1}=sprintf('%s %s %s', datestr(datetime, 'HH:MM:SS'), message_tag, message);
handles.eLog.String = str;
if is_save
  AppendLog(handles);
  ReadLog(handles);
end

% --------------------------------------------------------------------
function pbLogQ_Callback(~, ~, handles)
AddMessage(handles, '#Qvalue', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogComment_Callback(~, ~, handles)
AddMessage(handles, '#comment', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogAnesthesia_Callback(~, ~, handles)
AddMessage(handles, '#anesth_iso', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function eLogTemp_Callback(~, ~, handles)
AddMessage(handles, '#temperature', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogBPM_Callback(~, ~, handles)
AddMessage(handles, '#bpm', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogLamp_Callback(~, ~, handles)
AddMessage(handles, '#lamp', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogBolus_Callback(~, ~, handles)
AddMessage(handles, '#inj_bolus', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogCont_Callback(~, ~, handles)
AddMessage(handles, '#inj_continuous', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogSignal_Callback(~, ~, handles)
AddMessage(handles, '#signal', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pushbutton31_Callback(~, ~, handles)
AddMessage(handles, '#cannulation_attempts', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function DrawTimeline(handles)
if isfield(handles, 'monitoring')
  cla(handles.axes1)
  x = []; y = [];
  switch handles.pmSelectDisplay.Value
    case 1
      if ~isempty(handles.monitoring.bpm)
        x = [handles.monitoring.bpm.time];
        y = [handles.monitoring.bpm.value];
      end
    case 2
      if ~isempty(handles.monitoring.temperature)
        x = [handles.monitoring.temperature.time];
        y = [handles.monitoring.temperature.value];
      end
    case 3
      if ~isempty(handles.monitoring.anesth_iso)
        x = [handles.monitoring.anesth_iso.time];
        y = [handles.monitoring.anesth_iso.value];
      end
    case 4
      if ~isempty(handles.monitoring.signal)
        x = [handles.monitoring.signal.time];
        y = [handles.monitoring.signal.value];
      end
  end
  plot(x,y,'.-','parent',handles.axes1); hold on
  axis tight
end

% --------------------------------------------------------------------
function pmSelectDisplay_Callback(~, ~, handles)
DrawTimeline(handles)

% --------------------------------------------------------------------
function ProcessEPRImage(handles, destination_folder, source, type)
Qvalue = handles.monitoring.Qvalue;
if isempty(Qvalue), return; end

Q = [Qvalue.value]; Q = Q(end);

fprintf('Processing %s\n', source);
try
  switch type
    case 'fid'
      pars = ProcessLoadScenario('PulseRecon.scn', fullfile(handles.location.Matlab, 'epri\Scenario\IMRT\Pulse Fiducials trigger delay -2us.par'));
      pars.prc.save_data = 'yes';
      pars.fft.profile_correction = 'library';
      pars.fft.library_location = fullfile(handles.location.Matlab, 'calibration\cavity_profile');
      pars.fbp.Q = Q;
      ese_fbp(source, '', destination_folder, pars);
    case 'pO2'
      pars = ProcessLoadScenario('PulseRecon.scn', fullfile(handles.location.Matlab, 'epri\Scenario\IMRT\Pulse T1inv MSPS.par'));
      %     pars = ProcessLoadScenario('PulseRecon.scn', 'z:\CenterMATLAB\epri\Scenario\IMRT\Pulse T1inv MSPS experimental.par');
      pars.prc.save_data = 'yes';
      pars.fft.profile_correction = 'library';
      pars.fft.library_location = fullfile(handles.location.Matlab, 'calibration\cavity_profile');
      pars.fbp.Q = Q;
      ese_fbp_InvRec(source, '', destination_folder, pars);
  end
catch
end
fprintf('Done. \n');

% --------------------------------------------------------------------
function mEditLog_Callback(hObject, eventdata, handles)
IMRTfolder = handles.eFolder.String;
[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
fid = fopen(fname,'r');
str = {};
if fid~=-1 %if the file doesn't exist ignore the reading code
  while ~feof(fid)
    ss = fgets(fid); ss = strtrim(ss);
    if length(ss) > 0
      str{end+1} = ss;
    end
  end
  fclose(fid);
end
res = listdlg('Name', 'Select the entry', ...
  'ListString',str, 'ListSize', [560, 400],'SelectionMode','single');

if ~isempty(res)
  % create backup
  copyfile(fname, [fname, '.backup']);
  switch hObject
    case handles.mEditLogDisable
      % replace hash tag
      editstring = str{res};
      str{res} = strrep(editstring, '#', '--#');
    case handles.mEditLogEnable
      % replace hash tag
      editstring = str{res};
      editstring = strrep(editstring, '-#', '#');
      str{res} = strrep(editstring, '-#', '#');
  end
  % write back
  fid = fopen(fname,'w');
  if fid~=-1
    for ii=1:length(str)
      fprintf(fid, '%s\n', str{ii});
    end
    fclose(fid);
  end
  ReadLog(handles);
end

% --------------------------------------------------------------------
function pbOpenDirectory_Callback(hObject, eventdata, handles)
winopen(handles.eProjectPath.String);

% --------------------------------------------------------------------
function uid = generate_guid
%GENERATE_GUID  Generate a globally unique ID.
%   UID = GENERATE_GUID creates a universally unique identifier (UUID).
%   A UUID represents a 128-bit value. For more information including
%   algorithms used to create UUIDs, see RFC 4122:
%   A Universally Unique IDentifier (UUID) URN Namespace,
%   section 4.2 "Algorithms for Creating a Time-Based UUID".
%   Copyright 2012 Changjiang Yang
%   my_last_name.cj@gmail.com
import java.util.UUID;
uid = char(UUID.randomUUID());
uid = uid(end-8:end);
return

% --- Executes on button press in pushbutton33.
function pushbutton33_Callback(hObject, eventdata, handles)

h2 = findall(groot,'Type','figure');
fig_text = ''; separator = '';
for ii=1:length(h2)
  if ~isempty(get(h2(ii), 'Number'))
    fig_text = [fig_text, sprintf('%s%d', separator, get(h2(ii), 'Number'))];
    separator = ',';
  elseif ~contains(get(h2(ii), 'Name'), 'ArbuzGUI') && ...
      ~contains(get(h2(ii), 'Name'), 'Project Viewer') && ...
      ~contains(get(h2(ii), 'Name'), 'IMRT_MAIN')
    fig_text = [fig_text, sprintf('%s%s', separator, get(h2(ii), 'Name'))];
    separator = ',';
  end
end

str = inputdlg({'Leave the message', 'Screenshot figure(s)'}, '#addfigure', [5;5], {'Type here', fig_text});

if ~isempty(str)
  IMRTfolder = handles.eFolder.String;
  mkdir(fullfile(handles.eProjectPath.String, IMRTfolder, 'Screenshots'))
  C = strsplit(str{2},',');

  figure_link = '@'; separator = '';
  for ii=1:length(C)
    id = generate_guid();
    figure_link = [figure_link, separator, id];
    separator = ',';

    if isnan(str2double(C{ii}))
      for jj=1:length(h2)
        if ~isempty(C{ii}) && isequal(C{ii},get(h2(jj), 'Name'))
          handle = h2(jj);
          break;
        end
      end
    else
      handle = str2double(C{ii});
    end
    set(handle, 'PaperPositionMode', 'auto');
    print(handle, '-djpeg', fullfile(handles.eProjectPath.String, IMRTfolder, 'Screenshots',[id,'.jpg']));
    %     screencapture(figs(ii), fullfile(handles.eProjectPath.String, IMRTfolder, 'Screenshots',[id,'.jpg']));
  end
  figure_link = [figure_link, '@'];

  message = sprintf('%s %s', figure_link, str{1});
  AddMessage(handles, '#addfigure',message, true);
end

% --- Executes on button press in pbWord.
function pbWord_Callback(hObject, eventdata, handles)

IMRTfolder = handles.eFolder.String;
FigureFolder='Screenshots';

[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
[str, handles.monitoring] = read_log_file(fname);

[ActXWord,WordHandle]=StartWord('');
ActXWord.Visible = 1;

Style='Heading 1'; %NOTE! if you are NOT using an English version of MSWord you get
% an error here. For Swedish installations use 'Rubrik 1'.
WordText(ActXWord,'Report',Style,[0,2]);%two enters after text

for ii=1:length(str)
  Style='Normal';
  WordText(ActXWord,str{ii},Style,[0,1]);%enter after text
  if contains(str{ii},'#addfigure')
    res = regexp(str{ii}, '\@(?<List>[\w,]+)\@', 'names');
    if ~isempty(res)
      C = strsplit(res.List,',');
      for jj=1:length(C)
        try
          fname = fullfile(handles.eProjectPath.String, IMRTfolder, FigureFolder, [C{jj},'.jpg']);
          ImageIntoWord(ActXWord, fname);
        catch err
        end
      end
    end
  end
end

% --------------------------------------------------------------------
function CheckData(find_results, message)
if isempty(find_results)
    warndlg(message, 'Image not found', 'modal')
    error(message);
end

% --------------------------------------------------------------------
function figure1_SizeChangedFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
