function varargout = DosimetryFilmGUI(varargin)
% DOSIMETRYFILMGUI MATLAB code for DosimetryFilmGUI.fig
%      DOSIMETRYFILMGUI, by itself, creates a new DOSIMETRYFILMGUI or raises the existing
%      singleton*.
%
%      H = DOSIMETRYFILMGUI returns the handle to a new DOSIMETRYFILMGUI or the handle to
%      the existing singleton*.
%
%      DOSIMETRYFILMGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in DOSIMETRYFILMGUI.M with the given input arguments.
%
%      DOSIMETRYFILMGUI('Property','Value',...) creates a new DOSIMETRYFILMGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before DosimetryFilmGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to DosimetryFilmGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help DosimetryFilmGUI

% Last Modified by GUIDE v2.5 11-Jun-2018 14:35:23

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @DosimetryFilmGUI_OpeningFcn, ...
  'gui_OutputFcn',  @DosimetryFilmGUI_OutputFcn, ...
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
function DosimetryFilmGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to DosimetryFilmGUI (see VARARGIN)

% Choose default command line output for DosimetryFilmGUI
handles.ini = [];
handles.output = hObject;
handles.BEV = [];

% load calibration
handles.ini.Directories.Calibration = 'z:\CenterMATLAB\calibration\film_calibration';
r  = dir(fullfile(handles.ini.Directories.Calibration, '*.mat'));
clist = {r.name};
handles.pmCalibrations.String = clist;

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = DosimetryFilmGUI_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function pbBrowse_Callback(hObject, eventdata, handles)

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'SourcePath', 'C:/');

[FileName,PathName] = uigetfile({'*.tif;*.tiff', 'Tiff files from scanner (*.tif; *.tiff)'},...
  'Load file', old_path);

if ~(isequal(FileName,0) || isequal(PathName,0))
  handles.eFileName.String = fullfile(PathName, FileName);
  
  handles.ini.Directories.SourcePath = PathName;
  guidata(hObject, handles);
end

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)

%load calibration
str  = handles.pmCalibrations.String;
item = handles.pmCalibrations.Value;
cal = load(fullfile(handles.ini.Directories.Calibration, str{item}));

%load file
[handles.im] = imread(handles.eFileName.String);
[handles.iminfo] = imfinfo(handles.eFileName.String);

% convert to dose
handles.dose=rgb_to_dose(handles.im,cal.rd,cal.gd,cal.bd,cal.d,1);

%load plug production dataset
handles.BEV = [];
set(handles.pmBeamNumber, 'String', '-');
if exist(handles.eFileNameBeamData.String, 'file') == 2
  disp('Loading production dataset ...');
  s1 = load(handles.eFileNameBeamData.String);
  str = cell(length(s1.Bev_masks), 1);
  for ii=1:length(s1.Bev_masks), str{ii}=num2str(ii); end
  set(handles.pmBeamNumber, 'String', str, 'Value', 1);
  handles.BEV = s1.Bev_masks;
  handles.Btype = s1.plan_param.Boost_or_antiboost;
end

guidata(hObject, handles);
DrawAll(handles);

% --------------------------------------------------------------------
function rbDoseMap_Callback(hObject, eventdata, handles)
DrawAll(handles);

% --------------------------------------------------------------------
function DrawAll(handles)
cla(handles.axes1);
ImageResolution = 0.025 * 1.25;
switch handles.uibuttongroup1.SelectedObject.UserData
  case 0 % original
    ImageResolution = 25.4 / 150;
    imagesc(handles.im, 'Parent', handles.axes1);
    axis(handles.axes1, 'image');
  case 1
    cla(handles.axes1);
    res = 25.4 / 150; % in [mm]
    ImageResolution = res;
%     dims = size(handles.dose);
%     x = (1:dims(2))' * res; x = x - mean(x);
%     y = (1:dims(1))' * res; y = y - mean(y);
    imagesc(handles.dose, 'Parent', handles.axes1);
    axis(handles.axes1, 'image');
    colorbar(handles.axes1);
    colormap(handles.axes1, 'jet');
    
%     hold(handles.axes1, 'on')
%     if ~isempty(handles.BEV)
%       aperture = handles.BEV{1}.Antiboost_map;
%       aperture = rot90(aperture,2);
%       resBeam = 0.025 * 1.25; % in [mm]
%       dims = size(aperture);
%       xbeam = (1:dims(2))' * resBeam; xbeam = xbeam - mean(xbeam);
%       ybeam = (1:dims(1))' * resBeam; ybeam = ybeam - mean(ybeam);
%       t = contourc(xbeam, ybeam, aperture, [0.5,0.5]);
%       plot_contourmatrix(t, handles.axes1);
%     end    
    axis(handles.axes1, 'tight');
    
    max_dose = max(handles.dose(:));
    histogram(handles.dose(handles.dose > max_dose*0.25), 50, 'Parent', handles.axes2);
    axis(handles.axes2, 'tight');
  case 2
    res = 25.4 / 150; % in [mm]
    ImageResolution = res;
    dims = size(handles.dose);
    x = (1:dims(2))' * res; x = x - mean(x);
    y = (1:dims(1))' * res; y = y - mean(y);
    
    if ~isempty(handles.BEV)
      BeamNumber = str2double(handles.pmBeamNumber.String{handles.pmBeamNumber.Value});
      switch handles.Btype
        case 'Boost'
          aperture = handles.BEV{BeamNumber}.Dilated_boost_map;
        case 'AntiBoost'
          aperture = handles.BEV{BeamNumber}.Antiboost_map;
      end
      aperture = rot90(aperture,2);
      resBeam = 0.025 * 1.25; % in [mm]
      dims = size(aperture);
      xbeam = (1:dims(2))' * resBeam; xbeam = xbeam - mean(xbeam);
      ybeam = (1:dims(1))' * resBeam; ybeam = ybeam - mean(ybeam);
      
      [X,Y] = meshgrid(x,y);
      [XBEAM,YBEAM] = meshgrid(xbeam,ybeam);
      aperture_dose = interp2(XBEAM, YBEAM, aperture, X,Y);
      % registration
      
      [optimizer,metric] = imregconfig('multimodal');
      optimizer.MaximumIterations = 1500;
      optimizer.Epsilon = 0.25e-6;
      
      the_dose = handles.dose;
      max_dose = max(the_dose(:));
      the_dose(the_dose < 0.1*max_dose) = 0;

      set(handles.figure1,'pointer','watch'); drawnow;
      tform = imregtform(aperture_dose,the_dose,'rigid',optimizer,metric);
      disp('tform.T');
      disp(tform.T)
      set(handles.figure1,'pointer','arrow')
      
      movingRegistered = imwarp(aperture_dose,tform,'OutputView',imref2d(size(the_dose)));
      imshowpair(movingRegistered,the_dose, 'Parent', handles.axes1)
    end
    max_dose = max(handles.dose(:));
    histogram(handles.dose(handles.dose > max_dose*0.25), 50, 'Parent', handles.axes2);
case 3
    if ~isempty(handles.BEV)
      aperture = handles.BEV{1}.Antiboost_map;
      aperture = rot90(aperture,2);
      imagesc(aperture, 'Parent', handles.axes1);
      axis(handles.axes1, 'tight');
    end
end

hold(handles.axes1, 'on')
plot([5,5+10/ImageResolution], [8,8], 'w', 'Parent', handles.axes1);
for ii=0:2:9
  plot([5+ii/ImageResolution,5+(ii+1)/ImageResolution], [16,16], 'w', 'Parent', handles.axes1);
end
axis(handles.axes1, 'tight');

% --------------------------------------------------------------------
function pbBrowseBeamData_Callback(hObject, eventdata, handles)

dirs     = safeget(handles.ini, 'Directories', []);
old_path = safeget(dirs, 'BeamDataPath', 'C:/');

[FileName,PathName] = uigetfile({'*.m;*.mat', 'All supported types (*.m; *.mat)'},...
  'Load file', old_path);

if ~(isequal(FileName,0) || isequal(PathName,0))
  handles.eFileNameBeamData.String = fullfile(PathName, FileName);
  
  handles.ini.Directories.BeamDataPath = PathName;
  guidata(hObject, handles);
end

function[] = plot_contourmatrix(C, hh)
% plot_countourmatrix - Plots a contour matrix c as returned from contour
%
%  plot_contourmatrix(C)
% Rev History:
% 06-04-12 Created
figure(gcf);
holdstate = ishold;
hold(hh, 'on')
i = 1;
while i<=length(C)
  lev = C(1,i);
  cnt = C(2,i);
  plot(C(1,i+(1:cnt)),C(2,i+(1:cnt)),'w', 'Parent', hh);
  i = i+cnt+1;
end
if ~holdstate
  hold(hh, 'off')
end

% test
% Z:\CenterProjects\IMRT DOSIMETRY\IMRT5a_BlockExposures_07June2018\IMRT5aBlkExp07June2018005.tif
% V:\data\Imagnet_data_16\09\160928\IMRT5_003_AAA\Plug_production_dataset.mat

% Z:\CenterProjects\IMRT DOSIMETRY\IMRT5a_BlockExposures_07June2018\IMRT5aBlkExp07June2018003.tif
% V:\data\Imagnet_data_16\10\161005\IMRT5_017_BBB\Plug_production_dataset.mat