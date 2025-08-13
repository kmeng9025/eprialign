function varargout = IMRT_MAINRPT(varargin)
% IMRT_MAINRPT MATLAB code for IMRT_MAINRPT.fig
%      IMRT_MAINRPT, by itself, creates a new IMRT_MAINRPT or raises the existing
%      singleton*.
%
%      H = IMRT_MAINRPT returns the handle to a new IMRT_MAINRPT or the handle to
%      the existing singleton*.
%
%      IMRT_MAINRPT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMRT_MAINRPT.M with the given input arguments.
%
%      IMRT_MAINRPT('Property','Value',...) creates a new IMRT_MAINRPT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before IMRT_MAINRPT_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to IMRT_MAINRPT_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help IMRT_MAINRPT

% Last Modified by GUIDE v2.5 22-Jul-2021 12:42:49

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @IMRT_MAINRPT_OpeningFcn, ...
  'gui_OutputFcn',  @IMRT_MAINRPT_OutputFcn, ...
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
function IMRT_MAINRPT_OpeningFcn(hObject, eventdata, handles, varargin)
if ~isempty(varargin) && strcmp(varargin{1}, 'ForceRedraw'), return; end
handles.output = hObject;
handles.hGUI   = varargin{1};

handles.color.correct = [1,1,1];
handles.color.incorrect = [1,0.85,0.85];
handles.color.inproject = [0.8,0.8,0.9];

set(handles.pmSelectExperiment, 'String', {'Select Experiment ...','IMRT6','IMRT7','IMRT7TCD','OER-Graded Normoxia'});
file_load = {...
  'MRIax', 'MRI_axial', 'MRI', '*.img', true;...
  'MRIsg','MRI_saggital','MRI','*.img', false;...
  'CT','CT','DICOM3D','*.dcm', true;...
  'EPRfid','FID','3DEPRI','*.mat', true;...
  'EPRpO2','pO2_2','PO2_pEPRI','*.mat', true;...
  'EPRAmp','Amp','AMP_pEPRI','*.mat', false;...
  'EPRpO2other','pO2_1','PO2_pEPRI','*.mat', false;...
  'EPRpO2other2','pO2_3','PO2_pEPRI','*.mat', false...
  };
handles.Files = file_load(:,1);

edit_width = 75;
edit_height = 3.1;
file_block_heght = 3.5;
proc_block_height = 2.1;
button_height = 1.9;
button_height2 = 2.2;
button_width  = 5.0;
button_width2 = 5.4;
file_control_callback = @(hObject,eventdata)IMRT_MAINRPT('Action_Callback',hObject,eventdata,guidata(hObject));
process_control_callback = @(hObject,eventdata)IMRT_MAINRPT('ActionProcess_Callback',hObject,eventdata,guidata(hObject));

root_dir = fileparts(which('ArbuzGUI'));
open_file = imread(fullfile(root_dir, 'images', 'open_file.JPG'));
run_script = imread(fullfile(root_dir, 'images', 'run_script.JPG'));

pos = handles.panFiles.Position;
for ii=1:length(handles.Files)
  vertical_position = pos(4) - ii*file_block_heght - 0.8;
  vertical_position_offset = vertical_position + 0.1*file_block_heght;
  handles.(handles.Files{ii}).edit   = uicontrol(handles.panFiles, 'style', 'edit',...
    'units','characters','position',[pos(3)-edit_width-2.0, vertical_position, edit_width, edit_height],...
    'value', 2,'max',5);
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
  handles.(handles.Files{ii}).type = file_load{ii,3};
  handles.(handles.Files{ii}).exts = file_load{ii,4};
  handles.(handles.Files{ii}).need_to_load = file_load{ii,5};
  handles.(handles.Files{ii}).justloaded = false;
  handles.(handles.Files{ii}).inproject = false;
end

processing_steps = {...
  'ProcessEPRImages','Reconstruct EPR images';
  'CreateRegistrationSequence','Create registration sequence';
  'UpdateLoadedTransformation', 'Update loaded images transformation';
  'SegmentMRIaxFiducials','Segment MRI axial image';
  'SegmentMRIsagFiducials','Segment MRI saggital image';
  'SegmentEPRfiducials','Segment EPR fiducial image';
  'SegmentCTfiducials','Segment CT image';
  'RegisterMRI','Register MRI';
  'RegisterCT','Register CT';
  'SegmentTumor','Segment tumor from MRI (dummy)';
  'Visualization', 'Visualization';
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
handles.SegmentMRIaxFiducials.options = struct('first_slice', 1, 'last_slice', 100, 'image_threshold', 0.06, 'fiducial_number', 4, 'fiducials_threshold', 0.25,...
  'largest_noise_object', 60, 'dilate_radius', 5, 'extract_outline', true);
handles.SegmentMRIsagFiducials.options = struct('first_slice', 1, 'last_slice', 1000, 'image_threshold', 0.08, 'fiducial_number', 4, 'fiducials_threshold', 0.35,...
  'largest_noise_object', 7, 'dilate_radius', 2, 'extract_outline', true);
handles.SegmentEPRfiducials.options = struct('first_slice', 1, 'last_slice', 1000,'image_threshold', 0.25, 'fiducial_number', 4, 'fiducials_threshold', 0.25,...
  'largest_noise_object', 65, 'dilate_radius', 2, 'extract_outline', false,...
  'look_for_expansion', false);
handles.SegmentCTfiducials.options = struct('first_slice', 130, 'last_slice', 300,'fiducials_number',4,...
  'fiducials_voxels',200,'animal_density_min',400,'animal_density_max',2000,...
  'noise_density_max', 450);
handles.RegisterMRI.options = struct('fiducial_number', 4);
handles.RegisterCT.options = struct('fiducial_number', 4);
handles.PrepareImageData.options = struct('export_data', 0);
handles.WholeFieldPlanning.options = struct('prescription_Gy', 49.9, 'Field_size_mm', 35, ...
  'noise_density_max', 450, 'bone_segmentation_retrofix', 0);

handles.BoostFieldDosePlanning.options = struct('boost_1_aboost_2', 1, 'beams', 2, 'prescription_Gy', 13, ...
  'boost_margin', 1.2, 'noise_density_max', 450, 'bone_segmentation_retrofix', 0);
handles.PlanTreatment.options = struct('boost_1_aboost_2', 1, 'beams', 2, 'prescription_Gy', 13, ...
  'boost_margin', 1.2, 'antiboost_margin', 0.6, 'Plug_size', '16', 'noise_density_max', 450, ...
  'antiboost_algorithm', 0);

set(handles.eProjectPath, 'String', fileparts(arbuz_get(handles.hGUI, 'FileName')));

handles.Hypoxia_inCT = [];
handles.PO2_inCT = [];
handles.monitoring.anesth_iso = [];
handles.monitoring.temperature = [];
handles.monitoring.bpm = [];
handles.monitoring.Qvalue = [];
handles.monitoring.signal = [];


handles.location.Matlab = 'd:\CenterMATLAB';
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = IMRT_MAINRPT_OutputFcn(hObject, ~, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pmSelectExperiment_Callback(hObject, eventdata, handles)

handles.PrepareImageData.options.use_mask='Tumor_out';
 
switch handles.pmSelectExperiment.Value
  case 2 % IMRT-6
    handles.WholeFieldPlanning.options.prescription_Gy = 49.9;
    handles.WholeFieldPlanning.options.Field_size_mm = 35;
    %
    handles.BoostFieldDosePlanning.options.beams = 2;
    handles.BoostFieldDosePlanning.options.prescription_Gy = 13;
    %
    handles.PlanTreatment.options.beams = 2;
    handles.PlanTreatment.options.prescription_Gy = 13;
    handles.PlanTreatment.options.boost_margin = 1.2;
    handles.PlanTreatment.options.antiboost_margin = 0.6;
    handles.PlanTreatment.options.antiboost_algorithm = 0;
    
    handles.BoostFieldDosePlanning.options.bone_segmentation_retrofix = 1;
    handles.WholeFieldPlanning.options.bone_segmentation_retrofix = 1;
    
    
    guidata(hObject, handles);
    AddMessage(handles, '#set_options','IMRT6',true);
  case 3 % IMRT-7
    whole_field_dose = 47.2;
    boost_dose       = 12.9;
      
    handles.WholeFieldPlanning.options.prescription_Gy = whole_field_dose;
    handles.WholeFieldPlanning.options.Field_size_mm = 35;
    %
    handles.BoostFieldDosePlanning.options.beams = 2;
    handles.BoostFieldDosePlanning.options.prescription_Gy = boost_dose;
    %
    handles.PlanTreatment.options.beams = 2;
    handles.PlanTreatment.options.prescription_Gy = boost_dose;
    handles.PlanTreatment.options.boost_margin = 1.2;
    handles.PlanTreatment.options.antiboost_margin = 0.6;
    handles.PlanTreatment.options.antiboost_algorithm = 0;
    
    handles.BoostFieldDosePlanning.options.bone_segmentation_retrofix = 1;
    handles.WholeFieldPlanning.options.bone_segmentation_retrofix = 0;
    
    guidata(hObject, handles);
    AddMessage(handles, '#set_options','IMRT7',true);
  case 4 % IMRT-7 TCD-50
    whole_field_dose = 0;
    boost_dose       = 12.9;

    % expect dose from operator
    res = inputdlg('Dose for whole field, Gy', 'Input dose', 1, {'0'});
    if ~isempty(res), whole_field_dose = str2double(res{1}); end
      
    handles.WholeFieldPlanning.options.prescription_Gy = whole_field_dose;
    handles.WholeFieldPlanning.options.Field_size_mm = 35;
    %
    handles.BoostFieldDosePlanning.options.beams = 2;
    handles.BoostFieldDosePlanning.options.prescription_Gy = boost_dose;
    %
    handles.PlanTreatment.options.beams = 2;
    handles.PlanTreatment.options.prescription_Gy = boost_dose;
    handles.PlanTreatment.options.boost_margin = 1.2;
    handles.PlanTreatment.options.antiboost_margin = 0.6;
    handles.PlanTreatment.options.antiboost_algorithm = 0;
    
    handles.BoostFieldDosePlanning.options.bone_segmentation_retrofix = 1;
    handles.WholeFieldPlanning.options.bone_segmentation_retrofix = 0;
    
    guidata(hObject, handles);
    AddMessage(handles, '#set_options','IMRT7',true);    
  case 5 % OER-Graded Normoxia
    normoxia_dose = 12;
    hypoxia_dose  = 35;

    % expect dose from operator
    res = inputdlg({'Dose for Hypoxia, Gy', 'Dose for Normoxia, Gy'}, 'Input dose', 1, {num2str(hypoxia_dose), num2str(normoxia_dose)});
    if ~isempty(res), hypoxia_dose = str2double(res{1}); end
    if ~isempty(res), normoxia_dose = str2double(res{2}); end
      
    Inverted_Boost = hypoxia_dose < normoxia_dose;
    
    handles.WholeFieldPlanning.options.prescription_Gy = min(hypoxia_dose, normoxia_dose);
    handles.WholeFieldPlanning.options.Field_size_mm = 35;
    %
    handles.BoostFieldDosePlanning.options.boost_1_aboost_2 = iff(Inverted_Boost, 3, 1);
    handles.BoostFieldDosePlanning.options.beams = 2;
    handles.BoostFieldDosePlanning.options.prescription_Gy = abs(hypoxia_dose - normoxia_dose);
    %
    handles.PlanTreatment.options.boost_1_aboost_2 = iff(Inverted_Boost, 3, 1);
    handles.PlanTreatment.options.prescription_Gy = abs(hypoxia_dose - normoxia_dose);
    handles.PlanTreatment.options.inverted_block = hypoxia_dose < normoxia_dose;
    handles.PlanTreatment.options.boost_margin = 0.9;
    handles.PlanTreatment.options.antiboost_margin = 0.1;
    handles.PlanTreatment.options.antiboost_algorithm = 0;
    
    handles.BoostFieldDosePlanning.options.bone_segmentation_retrofix = 1;
    handles.BoostFieldDosePlanning.options.boost_margin = 0.9;
    
    handles.WholeFieldPlanning.options.bone_segmentation_retrofix = 0;
    
    handles.PrepareImageData.options.use_mask='Tumor_OER';
    
    guidata(hObject, handles);
    AddMessage(handles, '#set_options','OEC1', true);    
end


% --------------------------------------------------------------------
function pbProjectDirectory_Callback(hObject, ~, handles)
[PathName] = uigetdir(uncell(get(handles.eProjectPath, 'String')));
if ~isequal(PathName, 0)
  set(handles.eProjectPath, 'String', PathName);
  PrepareThePlan(handles);
end

% --------------------------------------------------------------------
function pbLoadProject_Callback(hObject, ~, handles)
set(handles.eProjectPath, 'String', fileparts(arbuz_get(handles.hGUI, 'FileName')));

% clear out the prepared data
handles.Hypoxia_inCT = [];
handles.PO2_inCT = [];

PrepareThePlan(handles);

% --------------------------------------------------------------------
function Action_Callback(hObject, eventdata, handles)
for ii=1:length(handles.Files)
  cset = handles.(handles.Files{ii});
  if hObject == cset.load
    load_image(handles.hGUI, get(cset.edit, 'String'), cset.id, cset.type);
    handles.(handles.Files{ii}).justloaded = true;
    ReadFileName(handles, handles.Files{ii});
    handles.(handles.Files{ii}).inproject = true;
    if ~isempty(eventdata)
      ActionProcess_Callback(handles.UpdateLoadedTransformation.pb_run, eventdata, handles);
    end
    return;
  elseif hObject == cset.browse
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
function ActionProcess_Callback(hObject, eventdata, handles)
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
    jj=1;
    while exist([fname,num2str(jj),'.png'], 'file')
      im = imread([fname,num2str(jj),'.png']);
      figure('Name',[fname,num2str(jj),'.png'],'NumberTitle','off'); image(im);axis image; axis off;
      set(gca, 'Position',[0.02,0.02,0.96,0.96])
      jj = jj+1;
    end
    jj=1;
    while exist([fname,num2str(jj),'.fig'], 'file')
      open([fname,num2str(jj),'.fig']);
      jj = jj+1;
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
  case 'CT', res = findCTfolder(PathName);
  case 'EPRfid'
    res = findEPRFile(EPRPathName, 'fid');
    if exist(res, 'file') ~= 2 % file exists
      res = findEPRFile(PathName, 'fid');
    end
  case {'EPRpO2', 'EPRAmp'}
    res = findEPRFile(PathName, 'pO2', 2);
    if exist(res, 'file') ~= 2
      res = findEPRFile(EPRPathName, 'pO2', 2);
    end
  case 'EPRpO2other'
    res = findEPRFile(PathName, 'pO2', 1);
    if exist(res, 'file') ~= 2
      res = findEPRFile(EPRPathName, 'pO2', 3);
    end
  case 'EPRpO2other2'
    res = findEPRFile(PathName, 'pO2', 3);
    if exist(res, 'file') ~= 2
      res = findEPRFile(EPRPathName, 'pO2', 1);
    end
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
    im_list = arbuz_FindImage(handles.hGUI, im_list, 'InName', '_1', {'FileName'});
    if ~isempty(im_list), res = im_list{1}.FileName; end
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
function CT_file = findCTfolder(fpath)
folders = dirlist(fpath);
CT_file = fpath;
for ii=1:length(folders)
  CT_folder = fullfile(fpath, folders(ii).name);
  [is, fname] = isCTfolder(CT_folder);
  if is, CT_file = fname; break; end
end

% --------------------------------------------------------------------
function EPR_file = findEPRFile(fpath, im_type, im_number)
EPR_file = fpath;
switch im_type
  case 'fid'
    files = dir(fullfile(fpath, '*3DHIRES*.mat'));
    if ~isempty(files)
      EPR_file = fullfile(fpath, files(end).name);
    end
  case 'fidraw'
    EPR_file = '';
    files = dir(fullfile(fpath, '*3DHIRES*.d01'));
    fnames = {files(:).name};
    for ii=1:length(fnames)
      EPR_file{ii} = fullfile(fpath, fnames{ii});
    end
  case 'pO2'
    files = dir(fullfile(fpath, '*MSPS_0p75*.mat'));
    fnames = {files(:).name};
    idx = cellfun( @(x) (x(1) == 'p'), fnames);
    files = files(idx);
    % sort files
    filenames = {files.name};
    filenames = sort(filenames);
    if ~isempty(files) && length(files) >= im_number
      EPR_file = fullfile(fpath, filenames{im_number});
    end
  case 'pO2raw'
    EPR_file = '';
    files = dir(fullfile(fpath, '*MSPS_0p75*.tdms'));
    fnames = {files(:).name};
    for ii=1:length(fnames)
      EPR_file{ii} = fullfile(fpath, fnames{ii});
    end
end

% --------------------------------------------------------------------
function [isCT, CT_path] = isCTfolder(fpath)
isCT = false; CT_path = fpath;
files = dir(fullfile(fpath, '*.dcm'));
if isempty(files), return; end
isCT = true;
CT_path = fullfile(fpath, files(1).name);

% --------------------------------------------------------------------
function load_image(hGUI, FileName, image_name, image_type)
set(hGUI,'Pointer','watch');drawnow
new_image = create_new_image(image_name, image_type, []);
new_image.FileName = FileName;
new_image.Name = image_name;
[new_image.data, new_image.data_info] = arbuz_LoadImage(new_image.FileName, new_image.ImageType);
new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
new_image.Aprime = eye(4);
set(hGUI,'Pointer','arrow');drawnow
arbuz_AddImage(hGUI, new_image);
arbuz_UpdateInterface(hGUI);

% --------------------------------------------------------------------
function [isEPR, EPR_path] = isEPRfolder(fpath)

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
      
      % check if fiducials image it is available
      res = findEPRFile(PathName, 'fid');
      if exist(res, 'file') ~= 2
        res = findEPRFile(EPRPathName, 'fidraw');
        for ii=1:length(res)
          if ~isempty(res)
            ProcessEPRImage(handles, PathName, res{ii}, 'fid');
          end
        end
      end
      
      % check if po2 images are available
      res = findEPRFile(EPRPathName, 'pO2raw');
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
    for ii=1:length(handles.Files)
      handles.(handles.Files{ii}).justloaded = false;
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
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % load T1 transformation for those files loaded
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.UpdateLoadedTransformation.checkbox.Value
    arbuz_ShowMessage(handles.hGUI, 'Setting T1 transformation to images.');
    arbuz_StageToTransformation(handles.hGUI, 'T2');
    for ii=1:length(handles.Files)
      if handles.(handles.Files{ii}).justloaded
        id = handles.(handles.Files{ii}).id;
        output_list = arbuz_FindImage(handles.hGUI, 'master', 'Name', id, {'Name', 'Anative'});
        arbuz_SetTransformation(handles.hGUI, 'T1', output_list{1}.Name, output_list{1}.Anative);
        handles.(handles.Files{ii}).justloaded = false;
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
    
    image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'InName', 'mri-fid-auto', {'data','Anative'});
    
    opts = handles.SegmentMRIaxFiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 1;
    opts.figure_filename=fname;
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      prange = res{ii}.range(3,:);
      prange = (prange - res{ii}.r0(3))./res{ii}.a(3);
      
      new_image.data(1,:) = res{ii}.r0+res{ii}.a*prange(1);
      new_image.data(2,:) = res{ii}.r0+res{ii}.a*prange(2);
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
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
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment EPR fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentEPRfiducials.checkbox.Value
    AddProcessingOptions(handles, 'SegmentEPRfiducials');
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist','data'});
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentEPRfiducials');
    
    [x,y,z] = meshgrid(1:128,1:128,1:128);
    EPR_MASK = false(128,128,128);
    EPR_MASK((x-64.5).^2+(y-64.5).^2+(z-64.5).^2 <= 42^2)=true;
    EPR_MASK((x-64.5).^2+(y-64.5).^2+(z-64.5).^2 > 34^2)=false;
    EPR_MASK((x-64.5).^2+(y-64.5).^2+(z-64.5).^2 <= 5^2)=false;
    opt = handles.SegmentEPRfiducials.options;
    opt.safety_mask = EPR_MASK;
    
    for ii=1:length(image_FID)
      [~, fiducials] = arbuz_fiducial_segmentation(image_FID{ii}.data, opt);
      new_image = create_new_image('epr-fid-auto','3DMASK',fiducials);
      arbuz_AddImage(handles.hGUI, new_image, image_FID{ii}.Image);
    end
    
    presentation = 2*fiducials;
    idx = any(any(presentation, 1),3);
    slice_range = find(idx,1,'first'):find(idx,1,'last');
    fig_opts.legend = 'green: fiducials; red: not assigned';
    fig_opts.show_min = 0.002;
    fig_opts.show_max = 0.005;
    fig_opts.slicedir = 2;
    figN = imrt_show_segmentation(1, image_FID{ii}.data, presentation, slice_range, fig_opts);
    set(figN, 'Position', get(0, 'Screensize'));
    epr_mkdir(fileparts([fname,'1.png']));
    saveas(figN, [fname,'1.png']);
    delete(figN);
    
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});
    
    opts = handles.SegmentEPRfiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 2;
    opts.figure_filename=fname;
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % segment CT fiducials
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.SegmentCTfiducials.checkbox.Value
    AddProcessingOptions(handles, 'SegmentCTfiducials');
    output_list = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'slavelist','data'});
    if ~isempty(output_list)
      opt = handles.SegmentCTfiducials.options;
      opt.figure = 1000;
      opt.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentCTfiducials');
      [animal, cast, mousebed, fiducials, bone] = arbuz_segment_CT(output_list{1}.data, opt);
      
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
    
    output_list = arbuz_FindImage(handles.hGUI, output_list, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'ct-fid-auto', {'data','Anative'});
    
    opts = handles.SegmentCTfiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 1;
    opts.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'SegmentCTfiducials');
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Register MRI Image
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.RegisterMRI.checkbox.Value
    AddProcessingOptions(handles, 'RegisterMRI');
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterMRI');
    
    % refit MRI fiducials
    output_list = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'InName', 'FID', {'data','Anative'});
    opts = handles.SegmentMRIaxFiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 1;
    opts.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterMRI');
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
    
    image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist'});
    output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', 'XYZ', {'data','Anative','Apre','A','Aprime'});
    
    % refit FID data
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});

    opts = handles.SegmentEPRfiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 3;
    opts.figure_filename=fname;
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end    
    % end refit
    
    output_list = arbuz_FindImage(handles.hGUI, 'master', 'InName', 'FID', {'slavelist'});
    output_list3 = arbuz_FindImage(handles.hGUI, output_list{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Anative'});
    
    opts.figure = -1;
    opts.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterMRI');
    
    opts.Aknown = hmatrix_rotate_y(90)*hmatrix_scale([1,-1,1]);
    [A1, err1] = arbuz_register_fiducials(output_list2, output_list3, opts);
    opts.Aknown = hmatrix_rotate_y(-90)*hmatrix_scale([1,-1,1]);
    [A2, err2] = arbuz_register_fiducials(output_list2, output_list3, opts);
    
    opts.figure = 2;
    if err1 < err2
      opts.Aknown = hmatrix_rotate_y(90)*hmatrix_scale([1,-1,1]);
      [A] = arbuz_register_fiducials(output_list2, output_list3, opts);
    else
      opts.Aknown = hmatrix_rotate_y(-90)*hmatrix_scale([1,-1,1]);
      [A] = arbuz_register_fiducials(output_list2, output_list3, opts);
    end
    
    arbuz_SetImage(handles.hGUI, image_MRIax, 'A', A);
    arbuz_SetImage(handles.hGUI, image_MRIax, 'Aprime', eye(4));
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Register CT Image
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.RegisterCT.checkbox.Value
    AddProcessingOptions(handles, 'RegisterCT');
    fname = fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterCT');

    % refit CT fiducials
    output_list = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'ct-fid-auto', {'data','Anative'});
    opts = handles.SegmentCTfiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 1;
    opts.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterCT');
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, output_list{1}.Image);
    end
    
    image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'slavelist'});
    output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'ImageType', 'XYZ', {'data','Anative','Apre','A','Aprime'});
    
    % refit FID data
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});

    opts = handles.SegmentEPRfiducials.options;
    opts.A = output_list{1}.Anative;
    opts.figure = 3;
    opts.figure_filename=fname;
    res = arbuz_fit_fiducials(output_list{1}.data, opts);
    
    new_image = create_new_image('','XYZ',[]);
    for ii=1:length(res)
      new_image.data = res{ii}.ends;
      new_image.Name = sprintf('AnFID%i',ii);
      arbuz_AddImage(handles.hGUI, new_image, image_FID{1}.Image);
    end    
    % end refit
    
    output_list = arbuz_FindImage(handles.hGUI, 'master', 'InName', 'FID', {'slavelist'});
    output_list3 = arbuz_FindImage(handles.hGUI, output_list{1}.SlaveList, 'ImageType', 'XYZ', {'data', 'Anative'});

    
    opts.figure = 2;
    opts.figure_filename=fullfile(handles.eProjectPath.String, IMRTfolder, 'RegisterCT');
    opts.Aknown = hmatrix_rotate_y(-90)*hmatrix_rotate_x(180);
    opts.fiducial_order = [4,3,2,1];
    %   A = opts.Aknown;
    [A] = arbuz_register_fiducials(output_list2, output_list3, opts);
    arbuz_SetImage(handles.hGUI, image_CT, 'A', A);
    arbuz_SetImage(handles.hGUI, image_CT, 'Aprime', eye(4));
    
    handles.Hypoxia_inCT = [];
    handles.PO2_inCT = [];
    guidata(hObject, handles);
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Create helper images for visualization
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.Visualization.checkbox.Value
    AddProcessingOptions(handles, 'Visualization');
    % MRI, Outline and fiducials
    if ~isempty(image_MRIax)
      output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-outline-autoSRF', {'Color'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-outline-auto', {'data','ASLAVE'});
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist','Anative'});
        output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-outline-autoSRF', {'Color'});
      end
      color = output_list2{1}.Color;
      color.EdgeColor = 'none';
      color.FaceAlpha = 0.35;
      color.FaceColor = [0.4940, 0.1840, 0.5560];
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);

      output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'I', 'mri-fid-autoSRF', {'Color'});
    if isempty(output_list2)
      % create surface image
      output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-fid-auto', {'data','ASLAVE'});
      [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
      new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
      new_image.A = output_list2{1}.Aslave;
      arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
      image_MRIax = arbuz_FindImage(handles.hGUI, image_MRIax, '', '', {'slavelist','Anative'});
      output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'mri-fid-autoSRF', {'Color'});
    end
    color = output_list2{1}.Color;
    color.EdgeColor = [0.75, 0, 0.75];
    color.FaceColor = 'none';
    arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
    arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end
    
    % EPR, Outline and fiducials
    if ~isempty(image_PO2)
      output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'epr-outline-autoSRF', {'Color'});
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
        output_list2 = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'epr-outline-autoSRF', {'Color'});
      end
      color = output_list2{1}.Color;
      color.EdgeColor = [0, 0.45, 0];
      color.FaceColor = 'none';
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end
    
    if ~isempty(image_FID)
      output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'Name', 'epr-fid-autoSRF', {'Color'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'Name', 'epr-fid-auto', {'data','ASLAVE'});
        [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
        new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
        new_image.A = output_list2{1}.Aslave;
        arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
        image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist','Anative'});
        output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'Name', 'epr-fid-autoSRF', {'Color'});
      end
      color = output_list2{1}.Color;
      color.EdgeColor = [0, 0.8, 0];
      color.FaceColor = 'none';
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
    end
    
    % CT fiducials
    if ~isempty(image_CT)
      output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'fid-autoSRF', {'Color'});
      if isempty(output_list2)
        % create surface image
        output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'fid-auto', {'data','ASLAVE'});
        if ~isempty(output_list2)
          [the_surface] = arbuz_mask2surface(output_list2{1}.data, [1,1,1]);
          new_image = create_new_image([output_list2{1}.Slave,'SRF'],'3DSURFACE',the_surface);
          new_image.A = output_list2{1}.Aslave;
          arbuz_AddImage(handles.hGUI, new_image, output_list2{1}.Image);
          image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'slavelist','Anative'});
          output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'fid-autoSRF', {'Color'});
        end
      end
      if ~isempty(output_list2)
        color = output_list2{1}.Color;
        color.EdgeColor = 'none';
        color.FaceColor = [0.8500, 0.3250, 0.0980];
        color.FaceAlpha = 0.2;
        arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
        arbuz_SetImage(handles.hGUI, output_list2, 'Visible', true);
      end
    end
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
  % Prepare image data
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if handles.PrepareImageData.checkbox.Value
    AddProcessingOptions(handles, 'PrepareImageData');
    image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'AShow','slavelist'});
    image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'data','slavelist'});
    image_PO2_tumor = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
    if isempty(image_PO2_tumor), error('Tumor mask was not found.'); end
    
    name_of_tumor_mask = handles.PrepareImageData.options.use_mask;
    image_PO2_tumor_out = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', name_of_tumor_mask, {'data', 'AShow'});
    if isempty(image_PO2_tumor_out)
      if strcmp(name_of_tumor_mask, 'Tumor_OER')
        % generate OER style addition
        se = strel('sphere',1);
        T = image_PO2_tumor{1}.data;
        H = image_PO2{1}.data <= 10;
        DRH = (imdilate(T, se) & ~T) & H; % hypoxic rim
        TRH = imdilate(T & H, se) & ~T; % hypoxia underlying rim
        Tumor_OER = T | (DRH & TRH);
        
        new_image = create_new_image('','3DMASK',[]);
        new_image.data = Tumor_OER;
        new_image.Name = name_of_tumor_mask;
        arbuz_AddImage(handles.hGUI, new_image, image_PO2{1}.Image);
        % Re-read pO2 and masks
        image_PO2 = arbuz_FindImage(handles.hGUI, image_PO2, '', '', {'data','slavelist'});
        image_PO2_tumor_out = arbuz_FindImage(handles.hGUI, image_PO2{1}.SlaveList, 'Name', name_of_tumor_mask, {'data', 'AShow'});
      else
        error(sprintf('pO2 image: %s mask was not found.', name_of_tumor_mask)); 
      end
    end
    
    res = arbuz_util_transform(handles.hGUI, image_PO2, image_CT{1}, []);
    handles.PO2_inCT = res.data;
    res = arbuz_util_transform(handles.hGUI, image_PO2_tumor, image_CT{1}, []);
    handles.Tumor_inCT = res.data;
    res = arbuz_util_transform(handles.hGUI, image_PO2_tumor_out, image_CT{1}, []);
    handles.TumorExt_inCT = res.data;
    
    hypoxia = image_PO2{1}.data <= 10 & image_PO2{1}.data > -25;
    res = arbuz_util_transform_data(handles.hGUI, image_PO2, double(hypoxia), true, image_CT{1}, []);
    handles.Hypoxia_inCT = res.data & handles.TumorExt_inCT;
    
    %    ibGUI((handles.PO2_inCT ~= -100) + handles.Tumor_inCT + handles.Hypoxia_inCT)
    %    ibGUI((handles.PO2_inCT <= 10 & handles.PO2_inCT >= 0) + handles.Tumor_inCT + handles.Hypoxia_inCT)

    handles.project_state = arbuz_get(handles.hGUI, 'state');
    guidata(hObject, handles);
    
    opts = handles.PrepareImageData.options;
    export_data = safeget(opts, 'export_data',0);
    if export_data
      [filename, pathname] = uiputfile({'*.mat'}, 'Save as');
      if ~isempty(filename)
        res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-cast-auto', {'data'});
        material_mask = res{1}.data*3;
        res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-outline-auto', {'data'});
        material_mask = material_mask + res{1}.data;
        res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-mousebed-auto', {'data'});
        material_mask = material_mask + res{1}.data*2;
        res = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'Name', 'ct-fid-auto', {'data'});
        material_mask = material_mask + res{1}.data*3;
        
        
        %  Tumor
        exp.Tumor = handles.TumorExt_inCT;
        %  Hypoxia
        exp.Hypoxia = handles.Hypoxia_inCT;
        % Leg segmentation
        exp.LegCTSegmentation = material_mask;
        
        % Beam Center
        exp.beam_center = epr_maskcm(handles.Hypoxia_inCT);
%         exp.beam_center = exp.beam_center([2,1,3]);

        save(fullfile(pathname, filename), '-struct', 'exp');
      end
    end
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
    
    if isempty(handles.Hypoxia_inCT) || isempty(handles.PO2_inCT) || project_state ~= handles.project_state
      % request reprocesssing
      handles.PrepareImageData.checkbox.Value = true;
      pbRunAll_Callback(hObject, [], handles);
      handles = guidata(handles.figure1);
    else

    opts = handles.BoostFieldDosePlanning.options;
    bone_segmentation_retrofix = safeget(opts, 'bone_segmentation_retrofix',0);

    image_CT = arbuz_FindImage(handles.hGUI, image_CT, '', '', {'data', 'AShow','slavelist'});
    
    Gantry_angles = 90 : -72 :-270+72;
    beam_center = epr_maskcm(handles.Hypoxia_inCT);
    beam_center = beam_center([2,1,3]);
    
    Bev_masks = cell(length(Gantry_angles), 1);
    for ii=1:length(Gantry_angles)
      Bev_masks{ii}.Angle = Gantry_angles(ii);
      [Bev_masks{ii}.Hypoxia, Bev_masks{ii}.Boost_bev_volume] = ...
        imrt_maskbev( Gantry_angles(ii), handles.Hypoxia_inCT, beam_center, []);
      [Bev_masks{ii}.Tumor] = ...
        imrt_maskbev( Gantry_angles(ii), handles.Tumor_inCT, beam_center, [] );
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
        imrt_maskbev( Gantry_angles(ii), handles.Hypoxia_inCT, beam_center, []);
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
      maskbev_depths(image_CT{1}.data, handles.Hypoxia_inCT, material_mask, Bev_masks, Gantry_angles, prescription, Center_postion );
    
    fpath = fullfile(handles.eProjectPath.String,IMRTfolder);
    fparts = strsplit(handles.eProjectPath.String, filesep);
    Experiment_name = fparts{end};
    
    switch  opts.boost_1_aboost_2
      case {1,3}, fname = sprintf('%s',Experiment_name,'_Boost');
      case 2, beamtimes = beamtimes * (12/7); % anti-boost
        fname = sprintf('%s',Experiment_name,'_AntiBoost');
    end
    
    %     Target = epr_maskcm(handles.Hypoxia_inCT);
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
    
    if isempty(handles.Hypoxia_inCT) || isempty(handles.PO2_inCT) || project_state ~= handles.project_state
      % request reprocesssing
      handles.PrepareImageData.checkbox.Value = true;
      pbRunAll_Callback(hObject, [], handles);
      handles = guidata(handles.figure1);
    else
      AddProcessingOptions(handles, 'PlanTreatment');
      
      WL_shift = Winston_Lutz_corrections();
      
      Target = handles.Hypoxia_inCT;
      beam_center = epr_maskcm(handles.Hypoxia_inCT);
      beam_center = beam_center([2,1,3]);
      %     beam_center = size(Target)/2;
      
      opts = handles.PlanTreatment.options;
      
      Plug_size = [num2str(safeget(opts, 'Plug_size', '16')), 'mm'];
      
      Gantry_angles = 90 : -72 :-270+72;
      for ii=1:length(Gantry_angles)
        Bev_masks{ii}.Angle = Gantry_angles(ii);
        [Bev_masks{ii}.Hypoxia, Bev_masks{ii}.Boost_bev_volume] = ...
          imrt_maskbev( Gantry_angles(ii), Target, beam_center, []);
        [Bev_masks{ii}.Tumor] = ...
          imrt_maskbev( Gantry_angles(ii), handles.TumorExt_inCT, beam_center, [] );
      end
      Bev_size = cellfun(@(x) x.Boost_bev_volume, Bev_masks);
      [~,min_idx] = min(Bev_size);
      Port_Gantry_Angle = Gantry_angles(min_idx);
      
      imfig = imrt_show_bev(Bev_masks, min_idx);
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
          imrt_maskbev( Gantry_angles(ii) , handles.TumorExt_inCT, beam_center, [] );
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
            
            imrt_openscad('beam', BEVTarget, Scale_factor, UVShift, Plug_size, filename);
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
            
            imrt_openscad('shell', BEVTarget, Scale_factor, UVShift, Plug_size, filename);
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
            
            imrt_openscad('inv-beam', BEVTarget, Scale_factor, UVShift, Plug_size, filename);
          end
          
          
      end
      
      imfig = imrt_show_bev(Bev_masks, -1);
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
    
    Target = handles.Hypoxia_inCT;
    Tumor  = handles.TumorExt_inCT;
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
    hypoxia_hit = numel(find((coverage_map > opts.prescription_Gy*0.9) & (handles.Hypoxia_inCT & handles.Tumor_inCT)));
    normoxia_hit = numel(find((coverage_map > opts.prescription_Gy*0.9) & (handles.TumorExt_inCT & ~handles.Hypoxia_inCT)));
    hypoxia_overall = numel(find(handles.Tumor_inCT & handles.Hypoxia_inCT));
    normoxia_overall = numel(find((handles.Tumor_inCT & ~handles.Hypoxia_inCT)));
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

% Assign files directory
PathName = handles.eProjectPath.String;
if ~isempty(PathName)
  for ii=1:length(handles.Files)
    fname = ReadFileName(handles, handles.Files{ii});
    if isempty(fname)
      fname = FindFileName(PathName, handles.Files{ii});
      set(handles.(handles.Files{ii}).edit, 'string', fname);
      handles.(handles.Files{ii}).inproject = false;
    else
      set(handles.(handles.Files{ii}).edit, 'string', fname);
      handles.(handles.Files{ii}).inproject = true;
    end
  end
end

% files to be loaded
for ii=1:length(handles.Files)
  fname = handles.(handles.Files{ii}).edit.String;
  handles.(handles.Files{ii}).checkbox.Value = 0;
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

isMriaxRegistration = false;

% image segmentation MRI axial
if ~isempty(image_MRIax)
  handles.SegmentMRIaxFiducials.checkbox.Value = isempty(image_MRIax{1}.SlaveList);
  if ~isempty(image_MRIax{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'Name', 'fid-auto', {});
    handles.SegmentMRIaxFiducials.checkbox.Value = isempty(output_list2);
  end
  handles.SegmentMRIaxFiducials.checkbox.Value = isempty(image_MRIax{1}.SlaveList);
  if ~isempty(image_MRIax{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', 'XYZ', {'color'});
    handles.SegmentMRIaxFiducials.checkbox.Value = isempty(output_list2);
    isMriaxRegistration = ~isempty(output_list2) && isequal(image_MRIax{1}.A, eye(4));
    if ~isempty(output_list2)
      color = output_list2{1}.Color;
      color.Color = 'red';
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      handles.SegmentMRIaxFiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
    if ~isMriaxRegistration
      handles.RegisterMRI.checkbox.BackgroundColor = handles.color.inproject;
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
    output_list2 = arbuz_FindImage(handles.hGUI, image_MRIsag{1}.SlaveList, 'InName', 'fid-auto', {'color'});
    handles.SegmentMRIsagFiducials.checkbox.Value = isempty(output_list2);
    if ~isempty(output_list2)
      color = output_list2{1}.Color;
      color.Color = 'red';
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      handles.SegmentMRIsagFiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% image segmentation FID EPR
if ~isempty(image_FID)
  handles.SegmentEPRfiducials.checkbox.Value = isempty(image_FID{1}.SlaveList);
  if ~isempty(image_FID{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', 'XYZ', {});
    output_list2 = arbuz_FindImage(handles.hGUI, output_list2, 'InName', 'AnFID', {'color'});
    handles.SegmentEPRfiducials.checkbox.Value = isempty(output_list2);
    if ~isempty(output_list2)
      color = output_list2{1}.Color;
      color.Color = 'green';
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      handles.SegmentEPRfiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% image segmentation CT
isCTRegistration = false;
if ~isempty(image_CT)
  handles.SegmentCTfiducials.checkbox.Value = isempty(image_CT{1}.SlaveList);
  if ~isempty(image_CT{1}.SlaveList)
    output_list2 = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'InName', 'AnFID', {'color'});
    handles.SegmentCTfiducials.checkbox.Value = isempty(output_list2);
    isCTRegistration = ~isempty(output_list2) && isequal(image_CT{1}.A, eye(4));
    if ~isempty(output_list2)
      color = output_list2{1}.Color;
      color.Color = 'magenta';
      arbuz_SetImage(handles.hGUI, output_list2, 'Color', color);
      handles.SegmentCTfiducials.checkbox.BackgroundColor = handles.color.inproject;
    end
    if ~isCTRegistration
      handles.RegisterCT.checkbox.BackgroundColor = handles.color.inproject;
    end
  end
end

% image registration
if isMriaxRegistration
  handles.RegisterMRI.checkbox.Value = true;
end

% image registration
if isCTRegistration
  handles.RegisterCT.checkbox.Value = true;
end

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
if ~isempty(handles.Hypoxia_inCT) && ~isempty(handles.PO2_inCT) &&  handles.project_state == project_state
  handles.PrepareImageData.checkbox.BackgroundColor = handles.color.inproject;
end

ReadLog(handles);

% --------------------------------------------------------------------
function ReadLog(handles)
IMRTfolder = handles.eFolder.String;
[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
fid = fopen(fname,'r');
str = {};
if fid~=-1 %if the file doesn't exist ignore the reading code
  handles.monitoring.anesth_iso = [];
  handles.monitoring.temperature = [];
  handles.monitoring.bpm = [];
  handles.monitoring.Qvalue = [];
  handles.monitoring.signal = [];
  while ~feof(fid)
    ss = fgets(fid); ss = strtrim(ss);
    if length(ss) > 0
      str{end+1} = ss;
      a = regexp(ss, '(?<date>\d\d:\d\d:\d\d)\s+(?<field>#\w+)\s+(?<value>[\d.eE]+)', 'names');
      try
        if ~isempty(a) && isfield(a, 'field')
          for ii=1:length(a)
            b = a(ii);
            switch b.field
              case '#anesth_iso'
                handles.monitoring.anesth_iso(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                handles.monitoring.anesth_iso(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#temperature'
                handles.monitoring.temperature(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                handles.monitoring.temperature(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#bpm'
                handles.monitoring.bpm(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                handles.monitoring.bpm(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#Qvalue'
                handles.monitoring.Qvalue(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                handles.monitoring.Qvalue(end).time  = sum(tt.*[1;1/60;1/3600]);
              case '#signal'
                handles.monitoring.signal(end+1).value = str2double(b.value);
                tt = sscanf(b.date, '%d:%d:%d');
                handles.monitoring.signal(end).time  = sum(tt.*[1;1/60;1/3600]);
            end
          end
        end
      catch
        a
      end
    end
  end
  fclose(fid);
end
set(handles.eLogOld,'String',str);
set(handles.eLog,'String',{})
guidata(handles.figure1, handles);

try
  % not every version of Java has jhEdit.anchorToBottom property
  jhEdit = findjobj(handles.eLog);
  jhEdit.anchorToBottom;
  jhEdit = findjobj(handles.eLogOld);
  jhEdit.anchorToBottom;
catch
end

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
function pbSaveLog_Callback(hObject, eventdata, handles)
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
function pbLogQ_Callback(hObject, eventdata, handles)
AddMessage(handles, '#Qvalue', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogComment_Callback(hObject, eventdata, handles)
AddMessage(handles, '#comment', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogAnesthesia_Callback(hObject, eventdata, handles)
AddMessage(handles, '#anesth_iso', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function eLogTemp_Callback(hObject, eventdata, handles)
AddMessage(handles, '#temperature', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogBPM_Callback(hObject, eventdata, handles)
AddMessage(handles, '#bpm', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogLamp_Callback(hObject, eventdata, handles)
AddMessage(handles, '#lamp', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogBolus_Callback(hObject, eventdata, handles)
AddMessage(handles, '#inj_bolus', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogCont_Callback(hObject, eventdata, handles)
AddMessage(handles, '#inj_continuous', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pbLogSignal_Callback(hObject, eventdata, handles)
AddMessage(handles, '#signal', ' ');
uicontrol(handles.eLog);

% --------------------------------------------------------------------
function pushbutton31_Callback(hObject, eventdata, handles)
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
function pmSelectDisplay_Callback(hObject, eventdata, handles)
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

% --- Executes on button press in pbOpenDirectory.
function pbOpenDirectory_Callback(hObject, eventdata, handles)
winopen(handles.eProjectPath.String);

