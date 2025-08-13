function varargout = MDACC_RPT(varargin)
% MDACC_RPT MATLAB code for MDACC_RPT.fig
%      MDACC_RPT, by itself, creates a new MDACC_RPT or raises the existing
%      singleton*.
%
%      H = MDACC_RPT returns the handle to a new MDACC_RPT or the handle to
%      the existing singleton*.
%
%      MDACC_RPT('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MDACC_RPT.M with the given input arguments.
%
%      MDACC_RPT('Property','Value',...) creates a new MDACC_RPT or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before MDACC_RPT_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to MDACC_RPT_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help MDACC_RPT

% Last Modified by GUIDE v2.5 19-Sep-2022 14:01:07

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @MDACC_RPT_OpeningFcn, ...
  'gui_OutputFcn',  @MDACC_RPT_OutputFcn, ...
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
function MDACC_RPT_OpeningFcn(hObject, ~, handles, varargin)
if ~isempty(varargin) && strcmp(varargin{1}, 'ForceRedraw'), return; end
handles.output = hObject;
handles.hGUI   = varargin{1};

handles.color.correct = [1,1,1];
handles.color.incorrect = [1,0.85,0.85];
handles.color.inproject = [0.8,0.8,0.9];

set(handles.pmSelectExperiment, 'String', {'Select Experiment ...','MDACC'});
file_load = {...
  'MRIax', 'MRI_axial', 'MRI', '*.img', true;...
%   'MRIsg','MRI_saggital','MRI','*.img', false;...
  'EPRcav','cavity','cavity','*.tdms', true;...
  'EPRfid','FID','3DEPRI','*.mat', true;...
  'EPRpO2','pO2_2','PO2_pEPRI','*.mat', true;...
  'EPRAmp','Amp','AMP_pEPRI','*.mat', false;...
  'EPRpO2other','pO2','PO2_pEPRI','*.mat', false;...
  %  'EPRpO2other2','pO2_3','PO2_pEPRI','*.mat', false...
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
file_control_callback = @(hObject,eventdata)MDACC_RPT('Action_Callback',hObject,eventdata,guidata(hObject));
process_control_callback = @(hObject,eventdata)MDACC_RPT('ActionProcess_Callback',hObject,eventdata,guidata(hObject));

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
handles.SegmentCTfiducials.options = struct('first_slice', 130, 'last_slice', 300,'fiducials_number',4,...
  'fiducials_voxels',200,'animal_density_min',400,'animal_density_max',2000,...
  'noise_density_max', 450);
handles.RegisterMRI.options = struct('fiducial_number', 4);
handles.RegisterCT.options = struct('fiducial_number', 4);


set(handles.eProjectPath, 'String', fileparts(arbuz_get(handles.hGUI, 'FileName')));

handles.Hypoxia_inCT = [];
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
function varargout = MDACC_RPT_OutputFcn(~, ~, handles)
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pmSelectExperiment_Callback(~, ~, handles)

switch handles.pmSelectExperiment.Value
  case 1 % MDACC
    AddMessage(handles, '#set_options','MDACC',true);
end

% --------------------------------------------------------------------
function pbProjectDirectory_Callback(~, ~, handles)
[PathName] = uigetdir(uncell(get(handles.eProjectPath, 'String')));
if ~isequal(PathName, 0)
  set(handles.eProjectPath, 'String', PathName);
  PrepareThePlan(handles);
  
  % update project name
  ini = arbuz_GetINI(handles.hGUI);
  ini.Directories.ProjectPath = PathName;
  arbuz_SetINI(handles.hGUI, ini);
  
  
end

% --------------------------------------------------------------------
function pbLoadProject_Callback(~, ~, handles)
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
    fname = get(cset.edit, 'String');
    [fpath,~,ext] = fileparts(fname);
    if contains(handles.Files{ii}, 'EPRpO2other')
      % load images different from already loaded
      flist = get(cset.edit, 'String');
      for jj=2:length(flist)
        [fpath,~,ext] = fileparts(flist{jj});
        if contains(ext,'mat')
          output_list = arbuz_FindImage(handles.hGUI, 'master', 'FileName', flist{jj}, {});
          if isempty(output_list)
            load_image(handles.hGUI, flist{jj}, [cset.id,'_o',num2str(jj)], cset.type);
          end
        else
          recon_path = which('PulseRecon.scn');
          [fpath] = fileparts(recon_path);
          scn = recon_path;
          par = fullfile(fpath, '\IMRT\Pulse T1inv MSPS experimental.par');
          [fields] = ProcessLoadScenario(scn, par);
          fields.fft.profile_file = fullfile(fpath, 'cavity_profile.mat');
          feval(fields.prc.process_method, flist{jj}, '', fpath, fields);
        end
      end
    elseif contains(cset.type, 'cavity')
      if contains(ext,'mat'), process_cavity_profile(fname); end
    elseif contains(ext,'tdms') 
      % Process images
      if contains(cset.type, '3DEPRI')
        scn = 'z:\CenterMATLAB\epri\Scenario\PulseRecon.scn';
        par = 'z:\CenterMATLAB\epri\Scenario\IMRT\Pulse Fiducials trigger delay -2us.par';
        [fields] = ProcessLoadScenario(scn, par);
        fields.fft.profile_file = fullfile(fpath, 'cavity_profile.mat'); 
        feval(fields.prc.process_method, fname, '', fpath, fields);
      elseif contains(cset.type, 'PO2')
        scn = 'z:\CenterMATLAB\epri\Scenario\PulseRecon.scn';
        par = 'z:\CenterMATLAB\epri\Scenario\IMRT\Pulse T1inv MSPS experimental.par';
        [fields] = ProcessLoadScenario(scn, par);
        fields.fft.profile_file = fullfile(fpath, 'cavity_profile.mat'); 
        feval(fields.prc.process_method, fname, '', fpath, fields);
      end
    else
      load_image(handles.hGUI, get(cset.edit, 'String'), cset.id, cset.type);
    end
    handles.(handles.Files{ii}).justloaded = true;
    ReadFileName(handles, handles.Files{ii});
    handles.(handles.Files{ii}).inproject = true;
    guidata(hObject, handles);
    PrepareThePlan(handles);
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
guidata(hObject, handles);

% --------------------------------------------------------------------
function res = FindFileName(PathName, file_type)
res = '';
EPRPathName = PathName;
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
  case 'EPRcav'
    res = findEPRFile(EPRPathName, 'profile', 1);
  case 'EPRfid'
    res = findEPRFile(EPRPathName, 'fid', 1);
  case {'EPRpO2', 'EPRAmp'}
    res = findEPRFile(PathName, 'pO2', 1);
  case 'EPRpO2other'
    res = findEPRFile(PathName, 'pO2raw', 100);
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

% this might be DICOM
if ~isMRI
  files = dir(fullfile(fpath, '*.dcm')); files(1:2)=[];
  if ~isempty(files)
    isMRI = true;
    MRI_file_path{1} = fullfile(files(1).folder, files(1).name);
  end
end

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

function result = find_data_file(fpath, template, file_type)
result = '';
files = dir(fullfile(fpath, [template, '.',file_type]));
if ~isempty(files)
  for jj=1:length(files)
    result{jj} = fullfile(fpath, files(jj).name);
  end
end

% --------------------------------------------------------------------
function EPR_file = findEPRFile(fpath, im_type, im_number)
EPR_file = fpath;
file_types = {'mat', 'd01', 'tdms'};

switch im_type
  case 'profile'
    for ii=1:length(file_types)
      EPR_file = find_data_file(fpath, '*profile*', file_types{ii});
      if ~isempty(EPR_file)
        break;
      end
    end
  case 'fid'
    for ii=1:length(file_types)
      EPR_file = find_data_file(fpath, '*image3D*', file_types{ii});
      if ~isempty(EPR_file)
        break;
      end
    end
  case 'pO2raw'
    EPR_file = find_data_file(fpath, 'p*image4D*', 'mat');
    EPR_file1 = find_data_file(fpath, '*image4D*', 'tdms');
    EPR_file2 = find_data_file(fpath, '*image4D*', 'd01');
    for ii=1:length(EPR_file1)
      [~, fname] = fileparts(EPR_file1{ii});
      isfound = false;
      for jj=1:length(EPR_file)
        if contains(EPR_file{jj},fname)
          isfound = true;
          break;
        end
      end
      if ~isfound, EPR_file = [EPR_file, EPR_file1(ii)]; end
    end
    for ii=1:length(EPR_file2)
      [~, fname] = fileparts(EPR_file2{ii});
      isfound = false;
      for jj=1:length(EPR_file)
        if contains(EPR_file{jj},fname)
          isfound = true;
          break;
        end
      end
      if ~isfound, EPR_file = [EPR_file, EPR_file2(ii)]; end
    end
  case 'pO2'
    EPR_file = find_data_file(fpath, 'p*image4D*', 'mat');
    if isempty(EPR_file)
      EPR_file1 = find_data_file(fpath, '*image4D*', 'tdms');
      EPR_file2 = find_data_file(fpath, '*image4D*', 'd01');
      EPR_file = [EPR_file1,EPR_file2];
    end
end
if im_number <= length(EPR_file)
  EPR_file = EPR_file{1};
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
if isempty(new_image.data) && contains(new_image.ImageType, 'MRI')
  [new_image.data, new_image.data_info] = arbuz_LoadImage(new_image.FileName, 'DICOM3D');
end

new_image.box = safeget(new_image.data_info, 'Bbox', size(new_image.data));
new_image.Anative = safeget(new_image.data_info, 'Anative', eye(4));
new_image.Aprime = eye(4);
set(hGUI,'Pointer','arrow');drawnow

if ~isempty(new_image.data)
  arbuz_AddImage(hGUI, new_image);
  arbuz_StageToTransformation(hGUI, 'T2');
  output_list = arbuz_FindImage(hGUI, 'master', 'Name', new_image.Name, {'Name', 'Anative'});
  arbuz_SetTransformation(hGUI, 'T1', output_list{1}.Name, output_list{1}.Anative);
  arbuz_UpdateInterface(hGUI);
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

  Status = arbuz_get(handles.hGUI, 'Status');

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
    disp('Refitting MRI');
    output_list = arbuz_FindImage(handles.hGUI, image_MRIax{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listMRI = arbuz_FindImage(handles.hGUI, output_list, 'InName', 'FID', {'data','Anative','A'});
    scale = hmatrix_scale([0.98, 0.98, 1]);
    [AMR, A2MR, rotMR_f4r, rotMR_f4r1] = f4model(output_listMRI{1}.data, output_listMRI{1}.Anative, ...
      handles.MRIf4Pars);

    disp('Refitting EPR');
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listEPR = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});
    [AEPR, A2EPR, rotEPR_f4r, rotEPR_f4r1] = f4model(output_listEPR{1}.data, output_listEPR{1}.Anative, ...
      handles.EPRf4Pars);

    arbuz_SetImage(handles.hGUI, image_MRIax, 'A', AMR \ AEPR);
    arbuz_SetImage(handles.hGUI, image_MRIax, 'Aprime', eye(4));
    Status('RegMRIax') = true;
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

    disp('Refitting CT');
    output_list = arbuz_FindImage(handles.hGUI, image_CT{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listCT = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'ct-fid-auto', {'data','Anative'});
    [ACT, A2CT] = f4model(output_listCT{1}.data, output_listCT{1}.Anative, ...
      handles.CTf4Pars);

    disp('Refitting EPR');
    image_FID = arbuz_FindImage(handles.hGUI, image_FID, '', '', {'slavelist'});
    output_list = arbuz_FindImage(handles.hGUI, image_FID{1}.SlaveList, 'ImageType', '3DMASK', {'slavelist'});
    output_listEPR = arbuz_FindImage(handles.hGUI, output_list, 'Name', 'epr-fid-auto', {'data','Anative'});
    [AEPR, A2EPR] = f4model(output_listEPR{1}.data, output_listEPR{1}.Anative, ...
      handles.EPRf4Pars);

    arbuz_SetImage(handles.hGUI, image_CT, 'A', ACT \ AEPR);
    arbuz_SetImage(handles.hGUI, image_CT, 'Aprime', eye(4));

    handles.Hypoxia_inCT = [];
    handles.PO2_inCT = [];
    guidata(hObject, handles);

    Status('RegCT') = true;
    arbuz_set(handles.hGUI, 'Status', Status);
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
    Status('Visual') = true;
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
    if contains(handles.Files{ii}, 'EPRpO2other')
      fname = FindFileName(PathName, handles.Files{ii});
      set(handles.(handles.Files{ii}).edit, 'string', fname, 'value', 1);
    elseif isempty(fname)
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
  elseif iscell(fname)
    if exist(fname{1}, 'file') ~= 2
      set(handles.(handles.Files{ii}).edit, 'background',handles.color.incorrect);
    else
      set(handles.(handles.Files{ii}).edit, 'background',handles.color.correct);
      handles.(handles.Files{ii}).checkbox.Value = handles.(handles.Files{ii}).need_to_load;
    end
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
if Status.isKey('Visual') && Status('Visual') == true
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
function pbSaveLog_Callback(~, ~, handles)
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
function mEditLog_Callback(hObject, ~, handles)
IMRTfolder = handles.eFolder.String;
[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
fid = fopen(fname,'r');
str = {};
if fid~=-1 %if the file doesn't exist ignore the reading code
  while ~feof(fid)
    ss = fgets(fid); ss = strtrim(ss);
    if ~isempty(ss)
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
function pbOpenDirectory_Callback(~, ~, handles)
winopen(handles.eProjectPath.String);


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
function pushbutton33_Callback(~, ~, handles)

h2 = findall(groot,'Type','figure');
fig_text = ''; separator = '';
for ii=1:length(h2)
  if ~isempty(get(h2(ii), 'Number'))
    fig_text = [fig_text, sprintf('%s%d', separator, get(h2(ii), 'Number'))];
    separator = ',';
  elseif ~contains(get(h2(ii), 'Name'), 'ArbuzGUI') && ...
      ~contains(get(h2(ii), 'Name'), 'Project Viewer') && ...
      ~contains(get(h2(ii), 'Name'), 'MDACC')
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
function pbWord_Callback(~, ~, handles)

IMRTfolder = handles.eFolder.String;
FigureFolder='Screenshots';

[~, exp] = fileparts(handles.eProjectPath.String);
fname = fullfile(handles.eProjectPath.String, IMRTfolder, [exp,'.log']);
[str, handles.monitoring] = read_log_file(fname);

[ActXWord]=StartWord('');
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

function process_cavity_profile(src_fname)
try
  switch fext
    case '.d01'
      [src.ax, src.y, src.dsc] = kv_d01read(src_fname);
    case '.tdms'
      [src.ax, src.y, src.dsc] = kv_smtdmsread(src_fname);
  end

  pars.pft = struct('data','0_','awin','ham','awidth',1,'aalpha',0.6,'ashift',0,'zerofill',2,'rshift',0,...
    'opt','imag','xshift',0,'lshift',-1);
  pars.echo_select_algorithm = 'unknown';
  pars.method = 1;
  pars.ShowFit = true;
  pars.BL = 2; pars.exclude = 1;
  [Amplitude, Phase, Frequency] = epri_CalculateAPF(src, pars);

  file_type = 'CavityProfile';
  [Frequency, idx] = sort(Frequency);
  Amplitude = Amplitude(idx);
  Phase     = Phase(idx);
  [fpath] = fileparts(src_fname);
  save(fullfile(fpath,'cavity_profile.mat'), 'file_type', 'Amplitude', 'Phase', 'Frequency');
catch
end
