function varargout = IMRTGUI(varargin)
% IMRTGUI MATLAB code for IMRTGUI.fig
%      IMRTGUI, by itself, creates a new IMRTGUI or raises the existing
%      singleton*.
%
%      H = IMRTGUI returns the handle to a new IMRTGUI or the handle to
%      the existing singleton*.
%
%      IMRTGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in IMRTGUI.M with the given input arguments.
%
%      IMRTGUI('Property','Value',...) creates a new IMRTGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before IMRTGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to IMRTGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help IMRTGUI

% Last Modified by GUIDE v2.5 21-Nov-2017 15:07:05

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @IMRTGUI_OpeningFcn, ...
  'gui_OutputFcn',  @IMRTGUI_OutputFcn, ...
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


% --- Executes just before IMRTGUI is made visible.
function IMRTGUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to IMRTGUI (see VARARGIN)

handles.options.use_cross_lines = false;
handles.options.use_hf10_contour = true;
handles.options.beam_alpha = false;
handles.options.beam_alpha_miss = 0.6;
handles.options.beam_contour = true;
handles.options.tumor_color = 'm';
handles.options.tumor_lw = 2;
handles.options.beam_color = 'k';
handles.options.beam_lw = 2;
handles.options.bev_color = 'r';
handles.BevAngleSelector.Value = 1;

% Choose default command line output for IMRTGUI
if nargin > 4
  handles.registration = varargin{2};
  [handles.path handles.name] = fileparts(handles.registration);
  %Check to see if there is a post processed dataset. Use that instead of the
  %supplied one.
  %   if exist(sprintf('%s',handles.path,filesep,'Post_processing_plug_dataset.mat'))
  %       disp('Found Post processed dataset. Loading data')
  %     handles = Load(handles,sprintf('%s',handles.path,filesep,'Post_processing_plug_dataset.mat'));
  %   else
  %   end
  handles = Load(hObject,handles, handles.registration);
  PlotAll(hObject,handles.axes1, handles, handles.options);
  handles.Boost_margin.Value = 1;
end
handles.output = hObject;

% handles.jScrollbar = javax.swing.JSlider;
% handles.jScrollbar.setOrientation(handles.jScrollbar.VERTICAL);
% javacomponent(handles.jScrollbar,[400,40,100,160]);
% set(handles.jScrollbar, 'Value',0, 'MajorTickSpacing', 5, 'PaintLabels',true, 'PaintTicks',true);
% set(handles.jScrollbar, 'Value',0, 'Minimum',-15, 'Maximum',15);
% hjSlider = handle(handles.jScrollbar, 'CallbackProperties');
% hjSlider.StateChangedCallback = ...
%   @(hjSlider,eventData) PlotAll(handles.axes1, guidata(handles.figure1));

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes IMRTGUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = IMRTGUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pbLoad.
function pbLoad_Callback(hObject, eventdata, handles)
% Load(handles, registration);
PlotAll(hObject,handles.axes1, handles, handles.options);

function handles = Load(hObject,handles, registration)

disp('Loading data ...');

% load the registration project
arbuz_OpenProject(handles.figure1, registration);
arbuz_ApplyTransformation(handles.figure1, '', 'fix');

% load IMRT production dataset
[fpath, ffile] = fileparts(registration);
try
  %Load the plug production dataset.
  handles.plugs = load(fullfile(fpath, 'Plug_production_dataset.mat'));
catch
  %if there is no plug production dataset, we have create one.
  %[ Coverage_stats ] = IMRT_n_port_planning( CT , EPRI, Tumor, plan_param )
  
end

iCT = arbuz_FindImage(handles.figure1, 'master', 'Name', 'CT', {'data'});
ipO2 = arbuz_FindImage(handles.figure1, 'master', 'InName', 'PO2', {'data'});

iCT{1}.Slave = 'pO2';
pars.dilate = 0;
% find image with 2 in it

for nI=1:length(ipO2)
  if ~isempty(strfind(ipO2{nI}.Image, '002')), break; end
end

fprintf('Using pO2 image %i\n', nI);
fprintf(sprintf('%s,','Using pO2 image ', ipO2{nI}.Image,'\n'));
fprintf('Transforming pO2 ...\n');
handles.pO2inCT = arbuz_util_transform(handles.figure1, ipO2{nI}, iCT{1}, pars);
fprintf(' done.\n');
handles.nBEV = 1;
%update the beam selection listbox on load of data
UpdateBeamSelectorList(hObject, handles, handles.plugs.Bev_masks)



set(handles.slOffset, 'Min', -5, 'Max', 5, 'value', 0);
set(handles.slOffset, 'SliderStep', [0.05,0.1]);

disp('Data Loaded');


function UpdateBeamSelectorList(hObject, handles, BevCell)
%BevCell should be the list of bevs the user can chose from.
List = {};
for ii= 1:length(BevCell)
  List{ii} = num2str(BevCell{ii}.Angle);
end
handles.BevAngleSelector.String = List;
% Update handles structure
guidata(hObject, handles);


function PlotAll(hObject,hax, handles, opt)

disp('Starting to render image + mask geometry')
tic
handles.LastRenderSuccessful = 0;

isboost = get(handles.pmBoost,'value') == 1;

use_hf10_contour = safeget(opt, 'use_hf10_contour', true);
use_cross_lines = safeget(opt, 'use_cross_lines', true);
beam_alpha = safeget(opt, 'beam_alpha', true);
beam_contour = safeget(opt, 'beam_contour', true);
beam_alpha_miss = safeget(opt, 'beam_alpha_miss', 0.5);
beam_alpha_fill = safeget(opt, 'beam_alpha_fill', false);
beam_fillcolor = safeget(opt, 'beam_fillcolor', [0.5,0.5,0.5]);
tumor_color = safeget(opt, 'tumor_color', 'm');
tumor_lw = safeget(opt, 'tumor_lw', 3);
beam_color = safeget(opt, 'beam_color', 'k');
bev_color = safeget(opt, 'bev_color', 'k');
beam_lw = safeget(opt, 'beam_lw', 2);

isboost = safeget(opt, 'isboost', isboost);

idx = false(size(handles.pO2inCT.data));
idx(1:300,1:300,1:300) = true;

pO2inCT = handles.pO2inCT.data(idx);
pO2inCT = reshape(pO2inCT,300,300,300);

axes(hax); cla(hax); hold(hax,'on');


nBev = handles.BevAngleSelector.Value;

switch handles.MaskSource.Value
  case 1
    %Set Boost beam to THEBEAM
    if ( sum(strfind(handles.Boost_margin.String{handles.Boost_margin.Value},'Margin')) == 0  && isboost)
      THEBEAM = handles.plugs.Bev_masks{nBev}.Dilated_boost_map;
    else
      THEBEAM = handles.plugs.Bev_masks{nBev}.Dilated_boost_map;
    end
    %Set AntiBoost beam to THEBEAM
    if ~isboost
      if isfield(handles.plugs.Bev_masks{nBev}, 'Antiboost_map')
        if sum(strfind(handles.Boost_margin.String{handles.Boost_margin.Value},'Margin')) == 0
          THEBEAM = handles.plugs.Bev_masks{nBev}.Custom_Dilated_Antiboost_map;
        else
          if ~any(handles.plugs.Bev_masks{nBev}.Antiboost_map(:))
            THEBEAM = handles.plugs.Bev_masks{nBev}.Antiboost_map;
          else
            if isfield(handles.plugs.Bev_masks{nBev}, 'Subtract_Dilated_boost_map')
              THEBEAM = handles.plugs.Bev_masks{nBev}.Subtract_Dilated_boost_map;
              [X,Y]=meshgrid(1:1500,1:1500);
              THEBEAM((X-750).^2 + (Y-750).^2 > (7/0.025)^2) = 1;
              THEBEAM = 1.0 - THEBEAM;
              disp('Warning! Antiboost mask was not found.');
            end
          end
        end
      else
        if isfield(handles.plugs.Bev_masks{nBev}, 'Subtract_Dilated_boost_map')
          THEBEAM = handles.plugs.Bev_masks{nBev}.Subtract_Dilated_boost_map;
          [X,Y]=meshgrid(1:1500,1:1500);
          THEBEAM((X-750).^2 + (Y-750).^2 > (7/0.025)^2) = 1;
        end
      end
    end
    
  case 2
    
    %if statement to check if we have this cashed, if so use
    %cashed dataset, otherwise generate.
    cached = safeget(handles,'Cashed_Generated_Bev_masks_parameters',struct('AntiboostAlgo',0,'BoostMargin',0, 'AngleSelect',0));
    
    if ~ (cached.AntiboostAlgo == handles.AntiboostChoiceMenu.Value && cached.BoostMargin == handles.Boost_margin.Value...
        && cached.AngleSelect == handles.BevAngleSelector.Value && handles.LastRenderSuccessful == 1 )
      %                 %assign THEBEAM
      %         switch handles.pmBoost.Value
      %             case 1
      %               THEBEAM = handles.Generated_Bev_masks{handles.nBEV}.Dilated_boost_map;
      %             case 2
      %               THEBEAM = handles.Generated_Bev_masks{handles.nBEV}.Antiboost_map;
      %         end
      
      
      %Generate New boost and antiboost beams.
      disp('Generating new boost or antiboost beam Geometry')
      
      handles.PossibleGeneratedAngles = [90:-72:(-270+72)];
      
      
      
      handles.GeneratedAngles = str2num(handles.BevAngleSelector.String{handles.BevAngleSelector.Value});
      
      
      PossibleAngles = {};
      for ii= 1:length( handles.PossibleGeneratedAngles)
        PossibleAngles{ii}.Angle = handles.PossibleGeneratedAngles(ii);
      end
      UpdateBeamSelectorList(hObject, handles, PossibleAngles)
      
      %handles.GeneratedAngles = [90]
      %Set the margins
      if ~(handles.Boost_margin.Value== 1)
        Boost_margin = str2num(handles.Boost_margin.String{handles.Boost_margin.Value});
        Antiboost_margin = Boost_margin/2;
      else
        Boost_margin = 1.2;  Antiboost_margin = Boost_margin/2;
      end
      BMargin = ((Boost_margin)/(1.26))+0.2;
      SE = strel('disk',floor(BMargin/0.025));
      AMargin = (((Antiboost_margin)*1.26))-0.2;
      SE_1 = strel('disk',floor(AMargin/0.025));
      
      %Boost
      [Generated_Bev_masks] = maskbev_single_angle( handles.GeneratedAngles , handles.plugs.CT_Frame_data.Tumor_CT, handles.plugs.CT_Frame_data.Hypoxia_CT, 1  );
      for ii = 1:length(handles.GeneratedAngles)
        Generated_Bev_masks{ii}.Dilated_boost_map = imdilate(Generated_Bev_masks{ii}.Boost_map,SE);
        Generated_Bev_masks{ii}.Subtract_Dilated_boost_map = imdilate(Generated_Bev_masks{ii}.Boost_map,SE_1);
      end
      %Only generate antiboost if user requests it.
      if handles.pmBoost.Value == 2
        
        %switch case for variant antiboosts
        switch handles.AntiboostChoiceMenu.Value
          case 1
            %Antiboost
            for ii = 1:length(handles.GeneratedAngles)
              [ Generated_Bev_masks{ii} ] = Find_antiboost_2D_mask_4_IMRTGUI_Version_2( Generated_Bev_masks{ii});
            end
          case 2
            
            [Generated_Tumor_Bev_masks] = maskbev_single_angle( handles.GeneratedAngles , handles.plugs.CT_Frame_data.Tumor_CT, handles.plugs.CT_Frame_data.Tumor_CT, 1  );
            
            for ii = 1:length(handles.GeneratedAngles)
              [ Generated_Bev_masks{ii} ] = Find_antiboost_2D_mask_5_IMRTGUI_Version( Generated_Bev_masks{ii},Generated_Tumor_Bev_masks{ii});
            end
          case 3
            disp('Not yet implemented, Prepare for errors.')
        end
        
        
      end
      
      handles.Generated_Bev_masks = Generated_Bev_masks;
      handles.Cashed_Generated_Bev_masks_parameters = struct('AntiboostAlgo', handles.AntiboostChoiceMenu.Value, 'BoostMargin', handles.Boost_margin.Value , 'AngleSelect', handles.BevAngleSelector.Value );
      
    end
    %assign THEBEAM
    disp('Use Cached Beam Geometry')
    switch handles.pmBoost.Value
      case 1
        THEBEAM = handles.Generated_Bev_masks{1}.Dilated_boost_map;
        handles.LastRenderSuccessful = 1;
      case 2
        THEBEAM = handles.Generated_Bev_masks{1}.Antiboost_map;
        if numel(find(THEBEAM))>0
          handles.LastRenderSuccessful = 1;
        end
    end
    
    
end

disp('Rotating and reslicing pO2 data')

szCT  = size(pO2inCT);
szBEV = size(handles.plugs.Bev_masks{1}.Boost_map);

planeOffset = get(handles.slOffset, 'value');
% planeOffset = get(handles.jScrollbar, 'value');

set(handles.eLog, 'string', sprintf('nBev=%i\nOff=%4.2f',nBev,planeOffset));

x = 0.1*linspace(1,szCT(1),szCT(1));
z = 0.1*linspace(1,szCT(3),szCT(3));

x1 = 0.025*linspace(1,szBEV(1),szBEV(1));
y1 = 0.025*linspace(1,szBEV(2),szBEV(2));

Mask = handles.plugs.Hypoxia_CT & handles.plugs.Tumor_CT;
Mask = Mask(idx);
Mask = reshape(Mask,300,300,300);
Tumor_CT = handles.plugs.Tumor_CT(idx);
Tumor_CT = reshape(Tumor_CT,300,300,300);

[Y,X,Z] = meshgrid(1:szCT(2),1:szCT(1),1:szCT(3));
treatment_center = [mean(X(Mask(:))),mean(Y(Mask(:))),mean(Z(Mask(:)))];
mag_factor = 1.26;

if use_cross_lines
  treatment_pixel = fix(treatment_center);
  pO2inCT(treatment_pixel(1)+(-1:1),treatment_pixel(2)+(-1:1),treatment_pixel(3)+(-1:1)) = 15000;
end

x = x - treatment_center(3)*0.1;
z = z - treatment_center(2)*0.1;
x2 = x1*mag_factor; x2 = x2 - x2(750);
y2 = y1*mag_factor; y2 = y2 - y2(750);

% plane
switch handles.MaskSource.Value
  case 1
    angle = 90-handles.plugs.Bev_masks{nBev}.Angle;
  case 2
    angle = 90-handles.Generated_Bev_masks{1}.Angle;
end
AA = hmatrix_translate([-treatment_center(2),-treatment_center(1),0])*...
  hmatrix_rotate_z(angle)* ...
  hmatrix_translate([treatment_center(2),treatment_center(1),0]);
[Ysl,Xsl,Zsl] = meshgrid(treatment_center(1)+planeOffset/0.1,1:szCT(1),1:szCT(3));
xyzvol = [Xsl(:) Ysl(:) Zsl(:)];
xyzvt = htransform_vectors(AA, xyzvol);
xt = reshape(xyzvt(:,1), size(Xsl));
yt = reshape(xyzvt(:,2), size(Ysl));
zt = reshape(xyzvt(:,3), size(Zsl));

% slTarget = squeeze(interp3(double(Mask),xt,yt,zt)) > 0.5;

% slCT = squeeze(interp3(handles.plugs.CT,xt,yt,zt));
% slCTrgb = arbuz_ind2rgb(slCT, 'bone', [0,4000]);
% hh00 = image(x, z, slCTrgb, 'Parent', hax);
% hh00 = imagesc(x,z,slCT/300,[0,40]);

slpO2 = squeeze(interp3(pO2inCT,xt,yt,zt));
rgb = arbuz_ind2rgb(slpO2, 'jet', [0,40]);

AlphaDataMask = imfill(slpO2 > 0,'holes');
AlphaData = AlphaDataMask;


if beam_alpha
  [X2,Y2] = meshgrid(x2,y2);
  [XX,YY] = meshgrid(x,z);
  BinCT = interp2(X2,Y2,THEBEAM,XX,YY) > 0.5;
  AlphaData = zeros(size(AlphaDataMask));
  AlphaData(AlphaDataMask) = beam_alpha_miss;
  AlphaData(BinCT&AlphaDataMask) = 1.0;
  if beam_alpha_fill
    AlphaData(BinCT) = 1.0;
    FillBeam = ~AlphaDataMask & BinCT;
    r = rgb(:,:,1); g = rgb(:,:,2); b = rgb(:,:,3);
    r(FillBeam) = beam_fillcolor(1); g(FillBeam) = beam_fillcolor(2); b(FillBeam) = beam_fillcolor(3);
    rgb(:,:,1) = r; rgb(:,:,2) = g; rgb(:,:,3) = b;
  end
end

hh0 = image(x,z,rgb,'parent',hax);
set(hh0, 'AlphaData', AlphaData, 'parent', hax)

if use_hf10_contour
  [~,hh00] = contour(x,z,slpO2,[10,10], 'linewidth', 2, 'parent', hax);
  set(hh00, 'color', 'w')
end

if beam_contour
  [~,hh04] = contour(x2,y2,THEBEAM,[0.5,0.5]);
  set(hh04, 'color', beam_color, 'linewidth',beam_lw);
end

% [~,hh01] = contour(x,x,slTarget,[0.5,0.5], 'linewidth', 3);
% set(hh01, 'color', 'r')
slTumor = squeeze(interp3(double(Tumor_CT),xt,yt,zt)) > 0.5;
[~,hh02] = contour(x,z,slTumor,[0.5,0.5], 'linewidth', tumor_lw, 'parent', hax);
set(hh02, 'color', tumor_color)
switch handles.MaskSource.Value
  case 1
    [~,hh03] = contour(x2,y2,handles.plugs.Bev_masks{nBev}.Boost_map,[0.5,0.5], 'linewidth', 1.5, 'parent', hax);
  case 2
    [~,hh03] = contour(x2,y2,handles.Generated_Bev_masks{1}.Boost_map,[0.5,0.5], 'linewidth', 1.5, 'parent', hax);
end

set(hh03, 'color', bev_color)

if use_cross_lines
  plot(x2,y2); plot(x2,-y2)
end

axis equal
axis([-15,15,-15,15])

%Set the angle string listing to whatever the current view is.
switch handles.MaskSource.Value
  case 1
    handles.BevAngleStr.String = num2str(handles.plugs.Bev_masks{handles.nBEV}.Angle);
  case 2
    handles.BevAngleStr.String = num2str(handles.Generated_Bev_masks{handles.nBEV}.Angle);
end

% Update handles structure
guidata(hObject, handles);

disp('Rendering is finished');
toc



function PlotBeam(hObject, eventdata, handles)
%Just Replot the beam with the new settings, keep the image unchanged.


% --- Executes when figure1 is resized.
function figure1_SizeChangedFcn(hObject, eventdata, handles)
pos = get(handles.figure1, 'position');

% %%BE orginal formatting
%
% set(handles.axes1, 'position', [20,50,pos(3)-100, pos(4)-60]);
% set(handles.pbLoad, 'position', [pos(3)-70,pos(4)-40,60, 30]);
%
% set(handles.pbPort1, 'position', [10,10,40,15]);
% set(handles.pbPort2, 'position', [pos(3)-100,10,40,15]);
%
% set(handles.slOffset, 'position', [pos(3)-25,10,15,150]);
% set(handles.eLog, 'position', [pos(3)-70,160,60, pos(4)-220]);
% set(handles.pmBoost, 'position', [pos(3)-70,pos(4)-60,60, 20]);

%New Formatting MM

set(handles.axes1, 'position', [20,50,pos(3)-120, pos(4)-60]);
axpos = get(handles.axes1, 'position');

% set(handles.BevAngleStr, 'position', [axpos(1)+5, axpos(2)+5,90,30]);
% set(handles.text2, 'position', [73,30,pos(3)-400,pos(4)-120]);
% set(handles.text3, 'position', [20,50,80, 60]);
% set(handles.text2, 'position', [73,30,pos(3)-400,pos(4)-120]);
% set(handles.text3, 'position', [73,30,pos(3)-400,pos(4)-120]);



set(handles.text2, 'position', [pos(3)-90,pos(4)-20,90, 15]);
set(handles.MaskSource, 'position', [pos(3)-90,pos(4)-50,90, 30]);
set(handles.text3, 'position', [pos(3)-90,pos(4)-55,90, 15]);
set(handles.AntiboostChoiceMenu, 'position', [pos(3)-90,pos(4)-80,90, 30]);
set(handles.text5, 'position', [pos(3)-90,pos(4)-90,90, 15]);
set(handles.BevAngleSelector, 'position', [pos(3)-90,pos(4)-120,90, 30]);


set(handles.pmBoost, 'position', [pos(3)-90,pos(4)-160,90,20]);
set(handles.pbLoad, 'position', [pos(3)-90,pos(4)-210,90,30]);
set(handles.Boost_margin, 'position', [pos(3)-90,pos(4)-190,90,30]);

% set(handles.pbPort1, 'position', [20,10,40,15]);
% set(handles.pbPort2, 'position', [pos(3)-100,10,40,15]);

set(handles.slOffset, 'position', [pos(3)-25,10,30,pos(4)-220]);
set(handles.eLog, 'position', [pos(3)-90,10,70, pos(4)-220]);



%Set GUI controls to control rel positoning of widgets.

% javacomponent(handles.jScrollbar, [pos(3)-25,10,15,150]);

% --------------------------------------------------------------------
function slOffset_Callback(hObject, eventdata, handles)
PlotAll(hObject,handles.axes1,handles,handles.options);

% --------------------------------------------------------------------
function pmBoost_Callback(hObject, eventdata, handles)
if sum(strfind(handles.Boost_margin.String{handles.Boost_margin.Value},'Margin')) == 0
  Boost_margin_Callback(hObject, eventdata, handles)
else
  PlotAll(hObject,handles.axes1, handles, handles.options);
end


% --------------------------------------------------------------------
function pbPort1_Callback(hObject, eventdata, handles)
handles.nBEV = max(handles.nBEV-1, 1);
guidata(hObject, handles);
PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function pbPort2_Callback(hObject, eventdata, handles)
handles.nBEV = min(handles.nBEV+1, length(handles.plugs.Bev_masks));
guidata(hObject, handles);
PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function popupmenu2_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function popupmenu2_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end

% --------------------------------------------------------------------
function Boost_margin_Callback(hObject, eventdata, handles)

% Hints: contents = cellstr(get(hObject,'String')) returns Boost_margin contents as cell array
%        contents{get(hObject,'Value')} returns selected item from Boost_margin

if sum(strfind(handles.Boost_margin.String{handles.Boost_margin.Value},'Margin')) == 0
  Margin = str2num(handles.Boost_margin.String{handles.Boost_margin.Value});
  disp(sprintf('%s','Setting Margin to ', handles.Boost_margin.String{handles.Boost_margin.Value}))
  
  BMargin = ((Margin)/(1.26))+0.2;
  SE = strel('disk',floor(BMargin/0.025));
  for ii = 1: length(handles.plugs.Bev_masks)
    handles.plugs.Bev_masks{ii}.Custom_Dilated_boost_map = imdilate(handles.plugs.Bev_masks{ii}.Boost_map,SE);
    handles.plugs.Bev_masks{ii}.Custom_Dilated_boost_margin = Margin;
  end
  
  switch handles.pmBoost.String{handles.pmBoost.Value}
    case 'Boost'
      disp('use Boost logic ')
      
      PlotAll(hObject,handles.axes1,handles, handles.options);
      guidata(hObject, handles);
    case 'Anti-boost'
      switch handles.MaskSource.Value
        case 1
          AMargin = (((Margin)*1.26)/2)-0.2;
          SE_1 = strel('disk',floor(AMargin/0.025));
          disp('use Antiboost logic ')
          for ii = 1: length(handles.plugs.Bev_masks)
            handles.plugs.Bev_masks{ii}.Custom_subtract_Dilated_boost_map = imdilate(handles.plugs.Bev_masks{ii}.Boost_map,SE_1);
            [ handles.plugs.Bev_masks{ii}] = Find_antiboost_2D_mask_4_IMRTGUI_Version( handles.plugs.Bev_masks{ii}   );
          end
          
          PlotAll(hObject,handles.axes1, handles, handles.options);
          guidata(hObject, handles);
          
        case 2
          PlotAll(hObject,handles.axes1, handles, handles.options);
          guidata(hObject, handles);
      end
      
  end
end

% --------------------------------------------------------------------
function Boost_margin_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end

% --------------------------------------------------------------------
function mView2PanelPlot_Callback(hObject, eventdata, handles)
options = [];
options.use_cross_lines = false;
options.use_hf10_contour = false;
options.beam_alpha = true;
options.beam_alpha_miss = 0.45;
options.beam_contour = true;
options.tumor_color = 'm';
options.tumor_lw = 1;
options.beam_color = 'k';
options.beam_lw = 1;
options.bev_color = 'r';
options.bev_lw = 1;
options.beam_alpha_fill = true;
options.beam_fillcolor = [0.75,0.75,0.75];

h = figure(500);
hax1 = subplot(1,2,1);
options.isboost = true;
PlotAll(hObject,hax1, handles, options);
hax2 = subplot(1,2,2);
options.isboost = false;
options.beam_alpha_miss = 0.35;
PlotAll(hObject,hax2, handles, options);

set(hax1, 'Position', [0.05, 0.1, 0.43, 0.9])
set(hax2, 'Position', [0.55, 0.1, 0.43, 0.9])


% --------------------------------------------------------------------
function mViewScheme1Panel_Callback(hObject, eventdata, handles)

idx = false(size(handles.pO2inCT.data));
idx(1:300,1:300,1:300) = true;

pO2inCT = handles.pO2inCT.data(idx);
pO2inCT = reshape(pO2inCT,300,300,300);

nBev = handles.nBEV;
szCT  = size(pO2inCT);
szBEV = size(handles.plugs.Bev_masks{nBev}.Boost_map);

planeOffset = get(handles.slOffset, 'value');

set(handles.eLog, 'string', sprintf('nBev=%i\nOff=%4.2f',nBev,planeOffset));

x = 0.1*linspace(1,szCT(1),szCT(1));
z = 0.1*linspace(1,szCT(3),szCT(3));

x1 = 0.025*linspace(1,szBEV(1),szBEV(1));
y1 = 0.025*linspace(1,szBEV(2),szBEV(2));

Mask = handles.plugs.Hypoxia_CT & handles.plugs.Tumor_CT;
Mask = Mask(idx);
Mask = reshape(Mask,300,300,300);
Tumor_CT = handles.plugs.Tumor_CT(idx);
Tumor_CT = reshape(Tumor_CT,300,300,300);

[Y,X,Z] = meshgrid(1:szCT(2),1:szCT(1),1:szCT(3));
treatment_center = [mean(X(Mask(:))),mean(Y(Mask(:))),mean(Z(Mask(:)))];
mag_factor = 1.26;

x = x - treatment_center(3)*0.1;
z = z - treatment_center(2)*0.1;
x2 = x1*mag_factor; x2 = x2 - x2(750);
y2 = y1*mag_factor; y2 = y2 - y2(750);

% plane
angle = 90-handles.plugs.Bev_masks{nBev}.Angle;
AA = hmatrix_translate([-treatment_center(2),-treatment_center(1),0])*...
  hmatrix_rotate_z(angle)* ...
  hmatrix_translate([treatment_center(2),treatment_center(1),0]);
[Ysl,Xsl,Zsl] = meshgrid(treatment_center(1)+planeOffset/0.1,1:szCT(1),1:szCT(3));
xyzvol = [Xsl(:) Ysl(:) Zsl(:)];
xyzvt = htransform_vectors(AA, xyzvol);
xt = reshape(xyzvt(:,1), size(Xsl));
yt = reshape(xyzvt(:,2), size(Ysl));
zt = reshape(xyzvt(:,3), size(Zsl));

slpO2 = squeeze(interp3(pO2inCT,xt,yt,zt));

AlphaDataMask = imfill(slpO2 > 0,'holes');

THEBEAMb = handles.plugs.Bev_masks{nBev}.Custom_Dilated_boost_map;
THEBEAMab = handles.plugs.Bev_masks{nBev}.Dilated_boost_map;

if isfield(handles.plugs.Bev_masks{nBev}, 'Antiboost_map')
  if sum(strfind(handles.Boost_margin.String{handles.Boost_margin.Value},'Margin')) == 0
    THEBEAMab = handles.plugs.Bev_masks{nBev}.Custom_Dilated_Antiboost_map;
  else
    if ~any(handles.plugs.Bev_masks{nBev}.Antiboost_map(:))
      THEBEAMab = handles.plugs.Bev_masks{nBev}.Antiboost_map;
    end
  end
end

figure(502); clf; hold on;
hh0 = imagesc(x,z,AlphaDataMask*0.6);
set(hh0, 'AlphaData', AlphaDataMask)

imm = zeros(size(THEBEAMb));
imm(THEBEAMab > 0.5) = 2;
imm(THEBEAMb > 0.5) = 1;
imm(THEBEAMab&THEBEAMb > 0.5) = 1.5;
hh1 = imagesc(x2,y2, imm);
set(hh1, 'AlphaData', THEBEAMb|THEBEAMab)

se = strel('disk',7);
ox = slpO2; ox(~imerode(AlphaDataMask, se)) = 100;
hypo = ox < 10.1;
contour(x,z,hypo, [0.5,0.5], 'linewidth', 2, 'color', 'm')

slTumor = squeeze(interp3(double(Tumor_CT),xt,yt,zt)) > 0.5;
[~,hh02] = contour(x,z,slTumor,[0.5,0.5], 'linewidth', 2, 'color', 'r');

legend({'Hypoxic areas', 'Tumor'});

axis equal
axis([-15,15,-15,15])
disp('Rendering is finished');


% --- Executes on selection change in MaskSource.
function MaskSource_Callback(hObject, eventdata, handles)
% hObject    handle to MaskSource (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns MaskSource contents as cell array
%        contents{get(hObject,'Value')} returns selected item from MaskSource

% switch handles.MaskSource.Value
%     case 1
%         %Return the masking process to the on disk record from "what was
%         %done"
%          PlotAll(hObject,handles.axes1, handles, handles.options);
%     case 2
%         %Generate the Bevs natively using the margin in margin box and the
%         %logic from antiboost. Then send to the plotting function.
%         PlotAll(hObject,handles.axes1, handles, handles.options);
switch handles.MaskSource.Value
  case 1
    handles.BevAngleSelector.Value = 1;
    
  case 2
end



% --- Executes during object creation, after setting all properties.
function MaskSource_CreateFcn(hObject, eventdata, handles)

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in AntiboostChoiceMenu.
function AntiboostChoiceMenu_Callback(hObject, eventdata, handles)


% --- Executes during object creation, after setting all properties.
function AntiboostChoiceMenu_CreateFcn(hObject, eventdata, handles)
% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function mExport2DMask_Callback(hObject, eventdata, handles)
blah = [];
for ii=1:length(handles.plugs.Bev_masks)
  blah{ii}.Hypoxia = handles.plugs.Bev_masks{ii}.Boost_map;
  %   blah{ii}.Boost = handles.plugs.Bev_masks{ii}.Dilated_boost_map;
  
  % generate tumor BEV
  [Generated_Tumor_Bev_masks] = ...
    maskbev_single_angle( handles.plugs.Bev_masks{ii}.Angle, ...
    handles.plugs.CT_Frame_data.Tumor_CT, handles.plugs.CT_Frame_data.Tumor_CT, 1  );
  blah{ii}.Tumor = Generated_Tumor_Bev_masks{1}.Boost_map;
  
  
  blah{ii}.AntiBoostProtection = handles.plugs.Bev_masks{ii}.Subtract_Dilated_boost_map;
  blah{ii}.AntiBoost = handles.plugs.Bev_masks{ii}.Antiboost_map;
end

fprintf('Variable tmp is created in the workspace.\n');
assignin('base', 'tmp', blah)
disp(blah);

% --------------------------------------------------------------------
function mExport3DMask_Callback(hObject, eventdata, handles)
