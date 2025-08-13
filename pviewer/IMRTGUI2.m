function varargout = IMRTGUI2(varargin)
% IMRTGUI2

% Edit the above text to modify the response to help IMRTGUI2

% Last Modified by GUIDE v2.5 29-Jul-2021 11:14:15

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
  'gui_Singleton',  gui_Singleton, ...
  'gui_OpeningFcn', @IMRTGUI2_OpeningFcn, ...
  'gui_OutputFcn',  @IMRTGUI2_OutputFcn, ...
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


% handles.plugs description
% handles.plugs.CTFrame - data in CT frame
% handles.plugs.CTFrame.Tumor
% handles.plugs.CTFrame.Hypoxia
% handles.plugs.CTFrame.MRIoutline
% handles.plugs.CTFrame.Bev_masks - cell stricture of BEVs
% BEV fields
% BEV.Angle     IMRT angle
% BEV.Hypoxia   Target
% BEV.Boost     Beam applied for treatment
% BEV.Antiboost Beam applied for treatment

% --------------------------------------------------------------------
function IMRTGUI2_OpeningFcn(hObject, eventdata, handles, varargin)

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

handles.options.CacheDirectory = 'C:/Temp/';
handles.pmBEVAngle.Value = 1;
handles.pmBoostMargin.Value = 1;

handles.drawing_method = 1;

handles.parameters.ImagePix = 0.1;    % mm
handles.parameters.PlugPix = 0.025;    % mm
handles.parameters.BoostMargin = 1.2;  % mm
handles.parameters.ABoostMargin = 0.6;  % mm
handles.parameters.Plane2PlugScale = 1.26;

handles.cache = [];

% Choose default command line output for IMRTGUI2
if nargin > 4
  handles.registration = varargin{2};
  [handles.path, handles.name] = fileparts(handles.registration);
  %Check to see if there is a post processed dataset. Use that instead of the
  %supplied one.
  %   if exist(sprintf('%s',handles.path,filesep,'Post_processing_plug_dataset.mat'))
  %       disp('Found Post processed dataset. Loading data')
  %     handles = Load(handles,sprintf('%s',handles.path,filesep,'Post_processing_plug_dataset.mat'));
  %   else
  %   end
  handles = Load(hObject,handles, handles.registration);
  PlotAll(hObject,handles.axes1, handles, handles.options);
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

% configure controls
set(handles.slOffset, 'Min', -5, 'Max', 5, 'value', 0);
set(handles.slOffset, 'SliderStep', [0.05,0.1]);

% UIWAIT makes IMRTGUI2 wait for user response (see UIRESUME)
% uiwait(handles.figure1);

% --------------------------------------------------------------------
function varargout = IMRTGUI2_OutputFcn(hObject, eventdata, handles)
% Get default command line output from handles structure
varargout{1} = handles.output;

% --------------------------------------------------------------------
function handles = Load(hObject,handles, registration)
% loader brings all imager to 300x300x300 size
registration_path = fileparts(registration);

% load blocks
par = [];
par.ProductionFolder = 'IMRT';
handles.plugs = load_aperture_data(registration_path, par);

fprintf('Loading registration project ...\n');
arbuz_OpenProject(handles.figure1, registration);
arbuz_ApplyTransformation(handles.figure1, '', 'fix');

parameters = [];
parameters.Verbosity = true;
parameters.MRIOutline = true;
parameters.LoadAllpO2 = true;
res = load_IMRTdata_from_registration(handles.figure1, parameters);

% resample all to 300x300x300
for ii=1:length(res.pO2_inCT)
  handles.plugs.CTFrame.pO2{ii} = res.pO2_inCT{ii}(1:300,1:300,1:300);
end
handles.plugs.nPO2 = res.nPO2;
set(handles.pmEPRimage, 'String', res.pO2list, 'Value', res.nPO2)
handles.plugs.CTFrame.Hypoxia = res.Hypoxia_inCT(1:300,1:300,1:300);
handles.plugs.CTFrame.Tumor = res.Tumor_inCT(1:300,1:300,1:300);
handles.plugs.CTFrame.MRIoutline = res.MRIoutline_inCT(1:300,1:300,1:300);

%update the beam selection listbox on load of data
UpdateBeamSelectorList(hObject, handles, handles.plugs.Bev_masks)
nBev        = handles.pmBEVAngle.Value;
planeOffset = get(handles.slOffset, 'value');
set(handles.eLog, 'string', sprintf('nBev=%i\nOff=%4.2f',nBev,planeOffset));

disp('Data are loaded.');

% --------------------------------------------------------------------
function [Boost] = MapHypoxia2Boost(Hypoxia, parameters)
pix          = safeget(parameters, 'PlugPix', 0.025);    % mm
margin       = safeget(parameters, 'BoostMargin', 1.2);  % mm
scale        = safeget(parameters, 'Plane2PlugScale', 1.26); % mm

margin_in_plug_coordinates = (margin/scale)+0.2;
SE = strel('disk',floor(margin_in_plug_coordinates/pix));

Boost = imdilate(Hypoxia,SE);

% --------------------------------------------------------------------
function [Protection] = MapHypoxia2Protection(Hypoxia, parameters)
pix          = safeget(parameters, 'PlugPix', 0.025);    % mm
margin       = safeget(parameters, 'ABoostMargin', 0.6);  % mm
scale        = safeget(parameters, 'Plane2PlugScale', 1.26); % mm

margin_in_plug_coordinates = margin/scale + 0.1;
SE = strel('disk',floor(margin_in_plug_coordinates/pix));

Protection = imdilate(Hypoxia,SE);

% --------------------------------------------------------------------
function im = get_from_cache(handles, field)
if isfield(handles.cache, field)
  im = handles.cache.(field);
  fprintf('Object %s is received from cache.\n', field)
else
  im = [];
end

% --------------------------------------------------------------------
function handles=send_to_cache(handles, field, im)
handles.cache.(field) = im;
guidata(handles.figure1, handles);
fprintf('Cache is updated.\n')

% --------------------------------------------------------------------
function UpdateBeamSelectorList(hObject, handles, BevCell)
%BevCell should be the list of bevs the user can chose from.
List = {};
for ii= 1:length(BevCell)
  List{ii} = num2str(BevCell{ii}.Angle);
end
handles.pmBEVAngle.String = List;
% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function PlotAll(hObject,hax, handles, opt)
do_not_use_cache = ~isequal(handles.mOtherUseCache.Checked,'on');
% do_not_use_cache = 1;

axes(hax); cla(hax); hold(hax,'on');

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
apertures = safeget(opt, 'apertures', []);
planeoffset = safeget(opt, 'planeoffset', 0);
drawing_method = safeget(opt, 'drawing_method', 1);

isboost = safeget(opt, 'isboost', isboost);

nBev        = handles.pmBEVAngle.Value;

Mask = handles.plugs.CTFrame.Hypoxia & handles.plugs.CTFrame.Tumor;
szCT  = size(Mask);
Tumor_CT = handles.plugs.CTFrame.Tumor;

[Y,X,Z] = meshgrid(1:szCT(2),1:szCT(1),1:szCT(3));
treatment_center = [mean(X(Mask(:))),mean(Y(Mask(:))),mean(Z(Mask(:)))];

szBEV = [1500 1500];

% Image coordinates
pars = handles.parameters;
x = pars.ImagePix*linspace(1,szCT(1),szCT(1));
z = pars.ImagePix*linspace(1,szCT(3),szCT(3));
x = x - treatment_center(3)*0.1;
z = z - treatment_center(2)*0.1;

x1 = pars.PlugPix*linspace(1,szBEV(1),szBEV(1));
y1 = pars.PlugPix*linspace(1,szBEV(2),szBEV(2));
x2 = x1*pars.Plane2PlugScale; x2 = x2 - x2(750);
y2 = y1*pars.Plane2PlugScale; y2 = y2 - y2(750);

angle = 90-handles.plugs.Bev_masks{nBev}.Angle;

AA = hmatrix_translate([-treatment_center(2),-treatment_center(1),0])*...
  hmatrix_rotate_z(angle)* ...
  hmatrix_translate([treatment_center(2),treatment_center(1),0]);
[Ysl,Xsl,Zsl] = meshgrid(treatment_center(1)+planeoffset/pars.ImagePix,1:szCT(1),1:szCT(3));
xyzvol = [Xsl(:) Ysl(:) Zsl(:)];
xyzvt = htransform_vectors(AA, xyzvol);
xt = reshape(xyzvt(:,1), size(Xsl));
yt = reshape(xyzvt(:,2), size(Ysl));
zt = reshape(xyzvt(:,3), size(Zsl));

% generate radiation beam target
mm = Tumor_CT & handles.plugs.CTFrame.Hypoxia;
goodpix=find(mm);
[igood,jgood,kgood]=ind2sub(size(mm),goodpix);
beam_target = [mean(jgood), mean(igood), mean(kgood)];

% oxygen image in the plane
if handles.drawing_method == 1
  nI = handles.plugs.nPO2;
  pO2code = sprintf('slpO2Ang%iOff%iI%i', handles.plugs.Bev_masks{nBev}.Angle, planeoffset, nI);
  pO2code(pO2code == '.' | pO2code == '-' | pO2code == '+') = '_';
  slpO2 = get_from_cache(handles, pO2code);
  if isempty(slpO2) || do_not_use_cache
    slpO2 = squeeze(interp3(handles.plugs.CTFrame.pO2{nI},xt,yt,zt));
    handles = send_to_cache(handles, pO2code, slpO2);
  end
end

% BEV of tumor contour
tumorcode = sprintf('TumorBEV%i', handles.plugs.Bev_masks{nBev}.Angle);
tumorcode(tumorcode == '.' | tumorcode == '-' | tumorcode == '+') = '_';
slTumor = get_from_cache(handles, tumorcode);
if isempty(slTumor) || do_not_use_cache
  [slTumor] = epr_maskbev(handles.plugs.Bev_masks{nBev}.Angle, Tumor_CT, beam_target);
  handles = send_to_cache(handles, tumorcode, slTumor);
end

% Leg outline
if drawing_method == 1
  legcode = sprintf('LegSliceAng%i', handles.plugs.Bev_masks{nBev}.Angle);
  legcode(legcode == '.' | legcode == '-' | legcode == '+') = '_';
  slLEG = squeeze(interp3(double(handles.plugs.CTFrame.MRIoutline),xt,yt,zt));
  handles = send_to_cache(handles, legcode, slLEG);
end

% Leg BEV
if 1 % needed for some algoritms
  legcode = sprintf('LegAng%i', handles.plugs.Bev_masks{nBev}.Angle);
  legcode(legcode == '.' | legcode == '-' | legcode == '+') = '_';
  slLEGBEV = get_from_cache(handles, legcode);
  if isempty(slLEGBEV) || do_not_use_cache
    if isfield(handles.plugs.CTFrame, 'MRIoutline')
      slLEGBEV = epr_maskbev( handles.plugs.Bev_masks{nBev}.Angle, ...
        handles.plugs.CTFrame.MRIoutline, beam_target);
    else
      slLEGBEV = true(szBEV);
    end
    
    handles = send_to_cache(handles, legcode, slLEGBEV);
  end
end

mymessage = "deflt";
mymessage2 = "";
switch handles.pmMaskSource.Value
  %-----------------------------------------------------------------------
  case 1 % Use masks from experiment
    mymessage = "src = experiment";
    slHypo = handles.plugs.Bev_masks{nBev}.Hypoxia;
    if isboost % Set Boost beam to BEAM
      slBEAM = handles.plugs.Bev_masks{nBev}.Boost;
      mymessage2 = sprintf('BEAM: %i',numel(find(slBEAM)));
    else % Set AntiBoost beam to BEAM
      if isfield(handles.plugs.Bev_masks{nBev}, 'Antiboost')
        slBEAM = double(handles.plugs.Bev_masks{nBev}.Antiboost);
      else
        slBEAM = zeros(szBEV);
      end
      S = numel(find(handles.plugs.Bev_masks{nBev}.Boost));
      mymessage2 = sprintf('BEAM: %i vs %i',numel(find(slBEAM)), S);
    end
    %-----------------------------------------------------------------------
  case 2 % GUI generated masks
    mymessage = "src = default algorithm";
    hypocode = sprintf('Hypo%iBM%4.2f', handles.plugs.Bev_masks{nBev}.Angle);
    hypocode(hypocode == '.' | hypocode == '-' | hypocode == '+') = '_';
    slHypo = get_from_cache(handles, hypocode);
    if isempty(slHypo) || do_not_use_cache
      mm = handles.plugs.CTFrame.Tumor & ...
        handles.plugs.CTFrame.Hypoxia;
      [slHypo, Hypo_volume] = epr_maskbev(handles.plugs.Bev_masks{nBev}.Angle, mm, beam_target);
      
      handles = send_to_cache(handles, hypocode, slHypo);
    end
    pars = handles.parameters;
    
    if isboost
      % get boost margin
      Boost_margin = 1.2;
      if ~(handles.pmBoostMargin.Value== 1)
        Boost_margin = str2double(handles.pmBoostMargin.String{handles.pmBoostMargin.Value});
      end
      pars.BoostMargin = Boost_margin;
      slBEAM = MapHypoxia2Boost(slHypo, pars);
      mymessage2 = sprintf('BEAM: %i',numel(find(slBEAM)));
    else
      Boost_margin = 1.2;
      if ~(handles.pmBoostMargin.Value== 1)
        Boost_margin = str2double(handles.pmBoostMargin.String{handles.pmBoostMargin.Value});
      end
      pars.ABoostMargin = Boost_margin/2;
      pars.BoostMargin = Boost_margin;
      
      %switch case for variant antiboosts
      abcode = sprintf('AB%iBM%4.2fABM%4.2', handles.plugs.Bev_masks{nBev}.Angle, pars.ABoostMargin);
      abcode(abcode == '.' | abcode == '-' | abcode == '+') = '_';
      boost = MapHypoxia2Boost(slHypo, pars);
      S = numel(find(boost));
      switch handles.AntiboostChoiceMenu.Value
        case 1 % Default antiboost emulation
          mymessage = "src = standard antiboost emulation";
          % an emulation of the default procedure
          slBEAM = get_from_cache(handles, ['M1', abcode]);
          if isempty(slBEAM) || do_not_use_cache
            hypoxia_protection = MapHypoxia2Protection(slHypo, pars);
            boost = MapHypoxia2Boost(slHypo, pars);
            [~,slBEAM] = expand(boost & ~hypoxia_protection, ~hypoxia_protection, S);
            handles = send_to_cache(handles, ['M1', abcode], slBEAM);
          end
          mymessage2 = sprintf('BEAM: %i vs %i',numel(find(slBEAM)), S);
        case 2
          error('Not yet implemented, Prepare for errors.')
          %           [Generated_Tumor_Bev_masks] = maskbev_single_angle( handles.GeneratedAngles , handles.plugs.CTFrame.Tumor_CT, handles.plugs.CTFrame.Tumor_CT, 1  );
          
          %           for ii = 1:length(handles.GeneratedAngles)
          %             [ Generated_Bev_masks{ii} ] = Find_antiboost_2D_mask_5_IMRTGUI_Version( Generated_Bev_masks{ii},Generated_Tumor_Bev_masks{ii});
          %           end
          %-----------------------------------------------------------------------
        case 3
          error('Not yet implemented, Prepare for errors.')
          % -------------------------------
        case 4 % max beam
          mymessage = "src = maximum tumor hit";
          slBEAM = get_from_cache(handles, ['M4', abcode]);
          if isempty(slBEAM) || do_not_use_cache
            hypoxia_protection = MapHypoxia2Protection(slHypo, pars);
            tumor = slTumor | hypoxia_protection;
            boost = MapHypoxia2Boost(slHypo, pars);
            
            x = 1:size(tumor,1);
            [X,Y] = meshgrid(x,x);
            
            S = numel(find(boost));
            
            % ab = contract(tumor, tumor & ~hypoxia_protection, S);
            
            CM = cm(X,Y,tumor);
            R = sqrt(S/pi);
            ab   = false(size(tumor));
            idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & ~hypoxia_protection & tumor;
            ab(idx_aboost) = true;
            all_protection = tumor & ~hypoxia_protection;
            [ismaxedout, ab] = expand(ab, all_protection, S);
            maxS = numel(find(all_protection));
            while ismaxedout && S < maxS
              R = R*1.5;
              idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
              ab(idx_aboost) = true;
              ab = contract(ab, all_protection, S);
              [ismaxedout, ab] = expand(ab, tumor & all_protection, S);
            end
            
            if numel(find(ab)) < S
              R = sqrt(S/pi);
              all_protection = slLEGBEV & ~hypoxia_protection;
              cc = zeros(size(all_protection));
              S1 = numel(find(ab));
              while S1 < S
                ab = ab | cc;
                R = R*1.1;
                idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
                cc(idx_aboost) = true;
                S1 = numel(find(ab | cc));
              end
            end
            [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
            if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
            slBEAM = ab;
            handles = send_to_cache(handles, ['M4', abcode], slBEAM);
          end
          mymessage2 = sprintf('BEAM: %i vs %i',numel(find(slBEAM)), S);
        case 5 % even more max
          mymessage = "src = maximum tumor hit";
          slBEAM = get_from_cache(handles, ['M5', abcode]);
          
          slHypo1 = bwmorph(slHypo, 'bridge');
          se = strel('disk',25);
          slHypo1 = imclose(slHypo1, se);
          
          protection_pars = pars;
          if isempty(slBEAM) || do_not_use_cache
            
            boost = MapHypoxia2Boost(slHypo, pars);
            x = 1:size(slHypo,1);
            [X,Y] = meshgrid(x,x);
            S = numel(find(boost));
            CM = cm(X,Y,slTumor | MapHypoxia2Protection(slHypo, pars));
            
            ABoostMargin = linspace(pars.ABoostMargin, 0.2, 5);
            for ii=1:length(ABoostMargin)
              protection_pars.ABoostMargin = ABoostMargin(ii);
              hypoxia_protection = MapHypoxia2Protection(slHypo1, protection_pars);
              ab = slTumor & ~hypoxia_protection;
              if numel(find(ab)) > S, break; end
            end
            
            if numel(find(ab)) < S
              R = sqrt(S/pi);
              all_protection = slLEGBEV & ~hypoxia_protection;
              cc = zeros(size(all_protection));
              S1 = numel(find(ab));
              while S1 < S
                ab = ab | cc;
                R = R*1.1;
                idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
                cc(idx_aboost) = true;
                S1 = numel(find(ab | cc));
              end
            end
            
            [~,ab] = expand(ab, all_protection, S);
            [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
            if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
            slBEAM = ab;
            handles = send_to_cache(handles, ['M5', abcode], slBEAM);
          end
          mymessage2 = sprintf('BEAM: %i vs %i (margin=%3.1f)',numel(find(slBEAM)), S, protection_pars.ABoostMargin);
          
          %             R = sqrt(S/pi);
          %             ab   = false(size(tumor));
          %             idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & ~hypoxia_protection & tumor;
          %             ab(idx_aboost) = true;
          %             all_protection = tumor & ~hypoxia_protection;
          %             [ismaxedout, ab] = expand(ab, all_protection, S);
          %             maxS = numel(find(all_protection));
          %             while ismaxedout && S < maxS
          %               R = R*1.5;
          %               idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
          %               ab(idx_aboost) = true;
          %               ab = contract(ab, all_protection, S);
          %               [ismaxedout, ab] = expand(ab, tumor & all_protection, S);
          %             end
          %
          %             if numel(find(ab)) < S
          %               R = sqrt(S/pi);
          %               all_protection = slLEGBEV & ~hypoxia_protection;
          %               cc = zeros(size(all_protection));
          %               S1 = numel(find(ab));
          %               while S1 < S
          %                 ab = ab | cc;
          %                 R = R*1.1;
          %                 idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
          %                 cc(idx_aboost) = true;
          %                 S1 = numel(find(ab | cc));
          %               end
          %             end
          %             [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
          %             if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
          %             slBEAM = ab;
          %             handles = send_to_cache(handles, ['M4', abcode], slBEAM);
          %           end
      end
    end
end

switch drawing_method
  case 1
    rgb = arbuz_ind2rgb(slpO2, 'jet', [0,40]);
    AlphaDataMask = imfill(slpO2 > 0,'holes');
    AlphaData = AlphaDataMask;
    
    if beam_alpha
      [X2,Y2] = meshgrid(x2,y2);
      [XX,YY] = meshgrid(x,z);
      BinCT = interp2(X2,Y2,slBEAM,XX,YY) > 0.5;
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
      [~,hh04] = contour(x2,y2,slBEAM,[0.5,0.5]);
      set(hh04, 'color', beam_color, 'linewidth',beam_lw);
      [~,hh04] = contour(x,z,slLEG,[0.5,0.5]);
      set(hh04, 'color', 'b', 'linewidth',0.5);
    end
    
    % [~,hh01] = contour(x,x,slTarget,[0.5,0.5], 'linewidth', 3);
    % set(hh01, 'color', 'r')
    slTumorPlane = squeeze(interp3(double(Tumor_CT),xt,yt,zt)) > 0.5;
    [~,hh02] = contour(x,z,slTumorPlane,[0.5,0.5], 'linewidth', tumor_lw, 'parent', hax);
    set(hh02, 'color', tumor_color)
    [~,hh03] = contour(x2,y2,slHypo,[0.5,0.5], 'linewidth', 1.5, 'parent', hax);
    set(hh03, 'color', bev_color);
    
    if use_cross_lines
      plot(x2,y2); plot(x2,-y2)
    end
    
    text(0.06, 0.92, sprintf('%s', mymessage), 'units', 'normalized', 'color', 'k');
    text(0.06, 0.84, sprintf('%s', mymessage2), 'units', 'normalized', 'color', 'k');
    axis(hax, 'tight')
    axis(hax, 'square')
    axis([-15,15,-15,15])
    %-------------------------------------------------------------------
  case 2 % BEV display in plugs space
    im = zeros(size(slLEGBEV));
    im(slLEGBEV > 0) = 1;
    im(slTumor > 0) = 2;
    if isboost
      im(slBEAM > 0) = 5;
      im = im - slHypo*0.5;
      im(1,1)=5; % color reference
    else
      im(slHypo > 0) = 3;
      tum = slBEAM & slTumor;
      im(slBEAM > 0) = 5;
      im = im - tum*0.5;
      im(1,1)=5; % color reference
    end
    imagesc(x1,y1,im,'parent',hax);
    colormap('jet')
    text(0.06, 0.92, sprintf('%s', mymessage), 'units', 'normalized', 'color', 'w');
    text(0.06, 0.86, sprintf('%s', mymessage2), 'units', 'normalized', 'color', 'w');
    axis(hax, 'tight')
    axis(hax, 'square')
  case 3
    rgb = arbuz_ind2rgb(slpO2, 'jet', [0,40]);
    AlphaDataMask = imfill(slpO2 > 0,'holes');
    AlphaData = AlphaDataMask;
    
    if beam_alpha
      [X2,Y2] = meshgrid(x2,y2);
      [XX,YY] = meshgrid(x,z);
      BinCT = interp2(X2,Y2,slBEAM,XX,YY) > 0.5;
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
      [~,hh04] = contour(x2,y2,slBEAM,[0.5,0.5]);
      set(hh04, 'color', beam_color, 'linewidth',beam_lw);
      [~,hh04] = contour(x,z,slLEG,[0.5,0.5]);
      set(hh04, 'color', 'b', 'linewidth',0.5);
    end
    
    % [~,hh01] = contour(x,x,slTarget,[0.5,0.5], 'linewidth', 3);
    % set(hh01, 'color', 'r')
    slTumorPlane = squeeze(interp3(double(Tumor_CT),xt,yt,zt)) > 0.5;
    [~,hh02] = contour(x,z,slTumorPlane,[0.5,0.5], 'linewidth', tumor_lw, 'parent', hax);
    set(hh02, 'color', tumor_color)
    [~,hh03] = contour(x2,y2,slHypo,[0.5,0.5], 'linewidth', 1.5, 'parent', hax);
    set(hh03, 'color', bev_color);
    
    if use_cross_lines
      plot(x2,y2); plot(x2,-y2)
    end
    
    text(0.06, 0.92, sprintf('%s', mymessage), 'units', 'normalized', 'color', 'k');
    text(0.06, 0.84, sprintf('%s', mymessage2), 'units', 'normalized', 'color', 'k');
    axis(hax, 'tight')
    axis(hax, 'square')
    axis([-15,15,-15,15])
end

for ii=1:length(apertures)
  circle2(0,0,apertures(ii)/2);
end
%Set the angle string listing to whatever the current view is.
% switch handles.pmMaskSource.Value
%   case 1
%     handles.BevAngleStr.String = num2str(handles.plugs.Bev_masks{handles.nBEV}.Angle);
%   case 2
%     handles.BevAngleStr.String = num2str(handles.Generated_Bev_masks{handles.nBEV}.Angle);
% end

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function PlotAll2(handles, hax, settings, plotopt)
do_not_use_cache = ~isequal(handles.mOtherUseCache.Checked,'on');
isboost = safeget(settings, 'boost', true);

nBev        = handles.pmBEVAngle.Value;
planeOffset = get(handles.slOffset, 'value');

szCT  = size(handles.plugs.CTFrame.Tumor);
Hypoxia = handles.plugs.CTFrame.Hypoxia & handles.plugs.CTFrame.Tumor;

axes(hax); cla(hax); hold(hax,'on');

[Y,X,Z] = meshgrid(1:szCT(2),1:szCT(1),1:szCT(3));
treatment_center = [mean(X(Hypoxia(:))),mean(Y(Hypoxia(:))),mean(Z(Hypoxia(:)))];
szBEV = [1500 1500];

% Image coordinates
pars = handles.parameters;
x = pars.ImagePix*linspace(1,szCT(1),szCT(1));
z = pars.ImagePix*linspace(1,szCT(3),szCT(3));
x = x - treatment_center(3)*0.1;
z = z - treatment_center(2)*0.1;

x1 = pars.PlugPix*linspace(1,szBEV(1),szBEV(1));
y1 = pars.PlugPix*linspace(1,szBEV(2),szBEV(2));
x2 = x1*pars.Plane2PlugScale; x2 = x2 - x2(750);
y2 = y1*pars.Plane2PlugScale; y2 = y2 - y2(750);

angle = 90-handles.plugs.Bev_masks{nBev}.Angle;

AA = hmatrix_translate([-treatment_center(2),-treatment_center(1),0])*...
  hmatrix_rotate_z(angle)* ...
  hmatrix_translate([treatment_center(2),treatment_center(1),0]);
[Ysl,Xsl,Zsl] = meshgrid(treatment_center(1)+planeOffset/pars.ImagePix,1:szCT(1),1:szCT(3));
xyzvol = [Xsl(:) Ysl(:) Zsl(:)];
xyzvt = htransform_vectors(AA, xyzvol);
xt = reshape(xyzvt(:,1), size(Xsl));
yt = reshape(xyzvt(:,2), size(Ysl));
zt = reshape(xyzvt(:,3), size(Zsl));

% generate radiation beam target
mm = handles.plugs.CTFrame.Tumor & ...
  handles.plugs.CTFrame.Hypoxia;
goodpix=find(mm);
[igood,jgood,kgood]=ind2sub(size(mm),goodpix);
beam_target = [mean(jgood), mean(igood), mean(kgood)];

% data mining
slpO2 = [];
slBEAM = [];
slLEGBEV = [];
slHypo = [];
slTumorBEV = [];
slTumor = [];
for ii=1:length(plotopt)
  if strcmp(plotopt{ii}.Source, 'oxygen') && isempty(slpO2)
    nI = handles.plugs.nPO2;
    pO2code = sprintf('slpO2Ang%iOff%iI%i', handles.plugs.Bev_masks{nBev}.Angle, planeOffset, nI);
    pO2code(pO2code == '.' | pO2code == '-' | pO2code == '+') = '_';
    slpO2 = get_from_cache(handles, pO2code);
    if isempty(slpO2) || do_not_use_cache
      slpO2 = squeeze(interp3(handles.plugs.CTFrame.pO2{nI},xt,yt,zt));
      handles = send_to_cache(handles, pO2code, slpO2);
    end
  end
  
  if strcmp(plotopt{ii}.Source, 'tumor') && isempty(slTumor)
    tumcode = sprintf('sltumAng%iOff%iI', handles.plugs.Bev_masks{nBev}.Angle, planeOffset);
    tumcode(tumcode == '.' | tumcode == '-' | tumcode == '+') = '_';
    slTumor = get_from_cache(handles, tumcode);
    if isempty(slTumor) || do_not_use_cache
      slTumor = squeeze(interp3(double(handles.plugs.CTFrame.Tumor),xt,yt,zt));
      handles = send_to_cache(handles, tumcode, slTumor);
    end
  end
  
  if strcmp(plotopt{ii}.Source, 'beamBEV') && isempty(slBEAM)
    mymessage = "deflt";
    mymessage2 = "";
    switch handles.pmMaskSource.Value
      %-----------------------------------------------------------------------
      case 1 % Use masks from experiment
        mymessage = "src = experiment";
        slHypo = handles.plugs.Bev_masks{nBev}.Hypoxia;
        if isboost % Set Boost beam to BEAM
          slBEAM = handles.plugs.Bev_masks{nBev}.Boost;
          mymessage2 = sprintf('BEAM: %i',numel(find(slBEAM)));
        else % Set AntiBoost beam to BEAM
          if isfield(handles.plugs.Bev_masks{nBev}, 'Antiboost')
            slBEAM = handles.plugs.Bev_masks{nBev}.Antiboost;
          else
            slBEAM = zeros(szBEV);
          end
          S = numel(find(handles.plugs.Bev_masks{nBev}.Boost));
          mymessage2 = sprintf('BEAM: %i vs %i',numel(find(slBEAM)), S);
        end
        %-----------------------------------------------------------------------
      case 2 % GUI generated masks
        mymessage = "src = default algorithm";
        hypocode = sprintf('Hypo%iBM%4.2f', handles.plugs.Bev_masks{nBev}.Angle);
        hypocode(hypocode == '.' | hypocode == '-' | hypocode == '+') = '_';
        slHypo = get_from_cache(handles, hypocode);
        if isempty(slHypo) || do_not_use_cache
          mm = handles.plugs.CTFrame.Tumor & ...
            handles.plugs.CTFrame.Hypoxia;
          [slHypo, Hypo_volume] = epr_maskbev(handles.plugs.Bev_masks{nBev}.Angle, mm, beam_target);
          
          handles = send_to_cache(handles, hypocode, slHypo);
        end
        pars = handles.parameters;
        
        if isboost
          % get boost margin
          Boost_margin = 1.2;
          if ~(handles.pmBoostMargin.Value== 1)
            Boost_margin = str2double(handles.pmBoostMargin.String{handles.pmBoostMargin.Value});
          end
          pars.BoostMargin = Boost_margin;
          slBEAM = MapHypoxia2Boost(slHypo, pars);
          mymessage2 = sprintf('BEAM: %i',numel(find(slBEAM)));
        else
          Boost_margin = 1.2;
          if ~(handles.pmBoostMargin.Value== 1)
            Boost_margin = str2double(handles.pmBoostMargin.String{handles.pmBoostMargin.Value});
          end
          pars.ABoostMargin = Boost_margin/2;
          pars.BoostMargin = Boost_margin;
          
          %switch case for variant antiboosts
          abcode = sprintf('AB%iBM%4.2fABM%4.2', handles.plugs.Bev_masks{nBev}.Angle, pars.ABoostMargin);
          abcode(abcode == '.' | abcode == '-' | abcode == '+') = '_';
          boost = MapHypoxia2Boost(slHypo, pars);
          S = numel(find(boost));
          switch handles.AntiboostChoiceMenu.Value
            case 1 % Default antiboost emulation
              mymessage = "src = standard antiboost emulation";
              % an emulation of the default procedure
              slBEAM = get_from_cache(handles, ['M1', abcode]);
              if isempty(slBEAM) || do_not_use_cache
                hypoxia_protection = MapHypoxia2Protection(slHypo, pars);
                boost = MapHypoxia2Boost(slHypo, pars);
                [~,slBEAM] = expand(boost & ~hypoxia_protection, ~hypoxia_protection, S);
                handles = send_to_cache(handles, ['M1', abcode], slBEAM);
              end
              slBEAM = double(slBEAM);
              mymessage2 = sprintf('BEAM: %i vs %i',numel(find(slBEAM)), S);
            case 2
              error('Not yet implemented, Prepare for errors.')
              %           [Generated_Tumor_Bev_masks] = maskbev_single_angle( handles.GeneratedAngles , handles.plugs.CTFrame.Tumor_CT, handles.plugs.CTFrame.Tumor_CT, 1  );
              
              %           for ii = 1:length(handles.GeneratedAngles)
              %             [ Generated_Bev_masks{ii} ] = Find_antiboost_2D_mask_5_IMRTGUI_Version( Generated_Bev_masks{ii},Generated_Tumor_Bev_masks{ii});
              %           end
              %-----------------------------------------------------------------------
            case 3
              error('Not yet implemented, Prepare for errors.')
              % -------------------------------
            case 4 % max beam
              mymessage = "src = maximum tumor hit";
              slBEAM = get_from_cache(handles, ['M4', abcode]);
              if isempty(slBEAM) || do_not_use_cache
                hypoxia_protection = MapHypoxia2Protection(slHypo, pars);
                tumor = slTumor | hypoxia_protection;
                boost = MapHypoxia2Boost(slHypo, pars);
                
                x = 1:size(tumor,1);
                [X,Y] = meshgrid(x,x);
                
                S = numel(find(boost));
                
                % ab = contract(tumor, tumor & ~hypoxia_protection, S);
                
                CM = cm(X,Y,tumor);
                R = sqrt(S/pi);
                ab   = false(size(tumor));
                idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & ~hypoxia_protection & tumor;
                ab(idx_aboost) = true;
                all_protection = tumor & ~hypoxia_protection;
                [ismaxedout, ab] = expand(ab, all_protection, S);
                maxS = numel(find(all_protection));
                while ismaxedout && S < maxS
                  R = R*1.5;
                  idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
                  ab(idx_aboost) = true;
                  ab = contract(ab, all_protection, S);
                  [ismaxedout, ab] = expand(ab, tumor & all_protection, S);
                end
                
                if numel(find(ab)) < S
                  R = sqrt(S/pi);
                  all_protection = slLEGBEV & ~hypoxia_protection;
                  cc = zeros(size(all_protection));
                  S1 = numel(find(ab));
                  while S1 < S
                    ab = ab | cc;
                    R = R*1.1;
                    idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
                    cc(idx_aboost) = true;
                    S1 = numel(find(ab | cc));
                  end
                end
                [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
                if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
                slBEAM = ab;
                handles = send_to_cache(handles, ['M4', abcode], slBEAM);
              end
              mymessage2 = sprintf('BEAM: %i vs %i',numel(find(slBEAM)), S);
            case 5 % even more max
              mymessage = "src = maximum tumor hit";
              slBEAM = get_from_cache(handles, ['M5', abcode]);
              
              slHypo1 = bwmorph(slHypo, 'bridge');
              se = strel('disk',25);
              slHypo1 = imclose(slHypo1, se);
              
              protection_pars = pars;
              if isempty(slBEAM) || do_not_use_cache
                
                boost = MapHypoxia2Boost(slHypo, pars);
                x = 1:size(slHypo,1);
                [X,Y] = meshgrid(x,x);
                S = numel(find(boost));
                CM = cm(X,Y,slTumor | MapHypoxia2Protection(slHypo, pars));
                
                ABoostMargin = linspace(pars.ABoostMargin, 0.2, 5);
                for ii=1:length(ABoostMargin)
                  protection_pars.ABoostMargin = ABoostMargin(ii);
                  hypoxia_protection = MapHypoxia2Protection(slHypo1, protection_pars);
                  ab = slTumor & ~hypoxia_protection;
                  if numel(find(ab)) > S, break; end
                end
                
                if numel(find(ab)) < S
                  R = sqrt(S/pi);
                  all_protection = slLEGBEV & ~hypoxia_protection;
                  cc = zeros(size(all_protection));
                  S1 = numel(find(ab));
                  while S1 < S
                    ab = ab | cc;
                    R = R*1.1;
                    idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
                    cc(idx_aboost) = true;
                    S1 = numel(find(ab | cc));
                  end
                end
                
                [~,ab] = expand(ab, all_protection, S);
                [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
                if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
                slBEAM = ab;
                handles = send_to_cache(handles, ['M5', abcode], slBEAM);
              end
              mymessage2 = sprintf('BEAM: %i vs %i (margin=%3.1f)',numel(find(slBEAM)), S, protection_pars.ABoostMargin);
              
              %             R = sqrt(S/pi);
              %             ab   = false(size(tumor));
              %             idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & ~hypoxia_protection & tumor;
              %             ab(idx_aboost) = true;
              %             all_protection = tumor & ~hypoxia_protection;
              %             [ismaxedout, ab] = expand(ab, all_protection, S);
              %             maxS = numel(find(all_protection));
              %             while ismaxedout && S < maxS
              %               R = R*1.5;
              %               idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
              %               ab(idx_aboost) = true;
              %               ab = contract(ab, all_protection, S);
              %               [ismaxedout, ab] = expand(ab, tumor & all_protection, S);
              %             end
              %
              %             if numel(find(ab)) < S
              %               R = sqrt(S/pi);
              %               all_protection = slLEGBEV & ~hypoxia_protection;
              %               cc = zeros(size(all_protection));
              %               S1 = numel(find(ab));
              %               while S1 < S
              %                 ab = ab | cc;
              %                 R = R*1.1;
              %                 idx_aboost = (X-CM(1)).^2+(Y-CM(2)).^2 <= R^2 & all_protection;
              %                 cc(idx_aboost) = true;
              %                 S1 = numel(find(ab | cc));
              %               end
              %             end
              %             [isupdate, conn] = connect_islands(ab, hypoxia_protection, CM);
              %             if isupdate, [~,ab] = expand(ab & ~conn, all_protection & ~conn, S); end
              %             slBEAM = ab;
              %             handles = send_to_cache(handles, ['M4', abcode], slBEAM);
              %           end
          end
        end
    end
  end
  
  if strcmp(plotopt{ii}.Source, 'legBEV') && isempty(slLEGBEV)
    legcode = sprintf('LegAng%i', handles.plugs.Bev_masks{nBev}.Angle);
    legcode(legcode == '.' | legcode == '-' | legcode == '+') = '_';
    slLEGBEV = get_from_cache(handles, legcode);
    if isempty(slLEGBEV) || do_not_use_cache
      if isfield(handles.plugs.CTFrame, 'MRIoutline')
        slLEGBEV = epr_maskbev( handles.plugs.Bev_masks{nBev}.Angle, ...
          handles.plugs.CTFrame.MRIoutline, beam_target);
      else
        slLEGBEV = true(szBEV);
      end
      
      handles = send_to_cache(handles, legcode, slLEGBEV);
    end
  end
  
  % BEV of tumor contour
  if strcmp(plotopt{ii}.Source, 'tumorBEV') && isempty(slTumorBEV)
    tumorcode = sprintf('TumorBEV%i', handles.plugs.Bev_masks{nBev}.Angle);
    tumorcode(tumorcode == '.' | tumorcode == '-' | tumorcode == '+') = '_';
    slTumorBEV = get_from_cache(handles, tumorcode);
    if isempty(slTumorBEV) || do_not_use_cache
      [slTumorBEV] = epr_maskbev(handles.plugs.Bev_masks{nBev}.Angle, handles.plugs.CTFrame.Tumor, beam_target);
      handles = send_to_cache(handles, tumorcode, slTumorBEV);
    end
  end

end

for ii=1:length(plotopt)
  switch plotopt{ii}.Method
    case 'none'
    case 'asis'
      range = safeget(plotopt{ii}, 'Range', [0 40]);
      rgb = arbuz_ind2rgb(slpO2, 'jet', range);
      AlphaDataMask = imfill(slpO2 > 0,'holes');
      hh0 = image(x,z,rgb,'parent',hax);
      set(hh0, 'AlphaData', AlphaDataMask, 'parent', hax)
    case 'alpha_outside_beam'
      range = safeget(plotopt{ii}, 'Range', [0 40]);
      rgb = arbuz_ind2rgb(slpO2, 'jet', range);
      AlphaDataMask = imfill(slpO2 > 0,'holes');
      
      beam_alpha_miss = safeget(plotopt{ii}, 'beam_alpha_miss', 0.5);
      beam_alpha_fill = safeget(plotopt{ii}, 'beam_alpha_fill', true);
      beam_fillcolor = safeget(plotopt{ii}, 'beam_fillcolor', [0.75,0.75,0.75]);
      
      [X2,Y2] = meshgrid(x2,y2);
      [XX,YY] = meshgrid(x,z);
      BinCT = interp2(X2,Y2,slBEAM,XX,YY) > 0.5;
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
      
      hh0 = image(x,z,rgb,'parent',hax);
      set(hh0, 'AlphaData', AlphaData, 'parent', hax)
    case 'contour'
      switch plotopt{ii}.Source
        case 'beamBEV', sl = slBEAM; ax1 = x2; ax2 = y2;
        case 'legBEV', sl = slLEGBEV; ax1 = x2; ax2 = y2; 
        case 'hypoxiaBEV', sl = slHypo; ax1 = x2; ax2 = y2; 
        case 'tumorBEV', sl = slTumorBEV; ax1 = x2; ax2 = y2; 
        case 'tumor', sl = slTumor; ax1 = x; ax2 = z; 
        case 'hypoxia', sl = slpO2 <= 10; ax1 = x; ax2 = z; 
      end
      sl_color = safeget(plotopt{ii}, 'color', 'k');
      sl_lw = safeget(plotopt{ii}, 'LW', 2);
      [~,hh04] = contour(ax1,ax2,sl,[0.5,0.5]);
      set(hh04, 'color', sl_color, 'linewidth',sl_lw);
    case 'area'
      switch plotopt{ii}.Source
        case 'beamBEV', sl = slBEAM; ax1 = x2; ax2 = y2;
        case 'legBEV', sl = slLEGBEV; ax1 = x2; ax2 = y2; 
        case 'hypoxiaBEV', sl = slHypo; ax1 = x2; ax2 = y2; 
        case 'tumorBEV', sl = slTumorBEV; ax1 = x2; ax2 = y2; 
        case 'hypoxia', sl = slpO2 <= 10; ax1 = x; ax2 = z; 
      end
      sl_color = epr_rgb(safeget(plotopt{ii}, 'color', 'k'));

      rgb = sl;
      rgb(:,:,1) = sl_color(1);
      rgb(:,:,2) = sl_color(2);
      rgb(:,:,3) = sl_color(3);
     
      hh0 = image(ax1,ax2,rgb,'parent',hax);
      set(hh0, 'AlphaData', sl > 0.5, 'parent', hax)
    case 'global'
      axis(hax, 'image');
      axis(hax, plotopt{ii}.range)
  end
end
  
% Update handles structure
guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function figure1_SizeChangedFcn(hObject, eventdata, handles)
pos = get(handles.figure1, 'position');

set(handles.axes1, 'position', [20,30,pos(3)-140, pos(4)-40]);
set(handles.panTools, 'position', [pos(3)-110,5,110, pos(4)-10]);

pos = get(handles.panTools, 'position');
set(handles.text6, 'position', [5,pos(4)-24,20, 15]);
set(handles.pmEPRimage, 'position', [25,pos(4)-20,75, 15]);
set(handles.text2, 'position', [5,pos(4)-40,95, 15]);
set(handles.pmMaskSource, 'position', [5,pos(4)-65,95, 30]);
set(handles.text5, 'position', [5,pos(4)-70,95, 15]);
set(handles.pmBEVAngle, 'position', [25,pos(4)-97,75, 30]);

set(handles.pmBoost, 'position', [5,pos(4)-110,95,20]);
set(handles.pmBoostMargin, 'position', [10,pos(4)-140,90,30]);
set(handles.AntiboostChoiceMenu, 'position', [10,pos(4)-160,90, 30]);
set(handles.pbLoad, 'position', [5,pos(4)-190,95,30]);

set(handles.slOffset, 'position', [75,10,25,pos(4)-210]);
set(handles.eLog, 'position', [5,10,70, pos(4)-210]);

% --------------------------------------------------------------------
function slOffset_Callback(hObject, eventdata, handles)
nBev        = handles.pmBEVAngle.Value;
planeOffset = get(handles.slOffset, 'value');
set(handles.eLog, 'string', sprintf('nBev=%i\nOff=%4.2f',nBev,planeOffset));
handles.options.planeoffset = planeOffset;
PlotAll(hObject,handles.axes1,handles,handles.options);

% --------------------------------------------------------------------
function pmBoost_Callback(hObject, eventdata, handles)
PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function pmBoostMargin_Callback(hObject, eventdata, handles)
PlotAll(hObject,handles.axes1, handles, handles.options);

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
options.apertures = 16:2:22;

h = figure;
hax1 = subplot(1,2,1);
options.isboost = true;
PlotAll(hObject,hax1, handles, options);
title(hax1, "Boost");
hax2 = subplot(1,2,2);
options.isboost = false;
options.beam_alpha_miss = 0.35;
PlotAll(hObject,hax2, handles, options);
title(hax2, "Anti-boost");

set(hax1, 'Position', [0.05, 0.1, 0.43, 0.9])
set(hax2, 'Position', [0.55, 0.1, 0.43, 0.9])

% --------------------------------------------------------------------
function mViewScheme1Panel_Callback(hObject, eventdata, handles)

pO2inCT = handles.plugs.pO2inCT;

nBev = handles.nBEV;
szCT  = size(pO2inCT);
planeOffset = get(handles.slOffset, 'value');

set(handles.eLog, 'string', sprintf('nBev=%i\nOff=%4.2f',nBev,planeOffset));

Tumor_CT = handles.plugs.CTFrame.Tumor;
Mask = handles.plugs.CTFrame.Hypoxia & Tumor_CT;

[Y,X,Z] = meshgrid(1:szCT(2),1:szCT(1),1:szCT(3));
treatment_center = [mean(X(Mask(:))),mean(Y(Mask(:))),mean(Z(Mask(:)))];

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
  if sum(strfind(handles.pmBoostMargin.String{handles.pmBoostMargin.Value},'Margin')) == 0
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

% --------------------------------------------------------------------
function pmMaskSource_Callback(hObject, eventdata, handles)
PlotAll(hObject,handles.axes1, handles, handles.options);
if handles.pmMaskSource.Value == 1
  handles.AntiboostChoiceMenu.Visible = 'off';
  handles.pmBoostMargin.Visible = 'off';
else
  handles.AntiboostChoiceMenu.Visible = 'on';
  handles.pmBoostMargin.Visible = 'on';
end

% --------------------------------------------------------------------
function pmEPRimage_Callback(hObject, eventdata, handles)
handles.plugs.nPO2 = handles.pmEPRimage.Value;
guidata(handles.figure1, handles);
PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function AntiboostChoiceMenu_Callback(hObject, eventdata, handles)
PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function mExport2DMask_Callback(hObject, eventdata, handles)
blah = [];
for ii=1:length(handles.plugs.Bev_masks)
  blah{ii}.Boost = handles.plugs.Bev_masks{ii}.Boost;
  %   blah{ii}.Boost = handles.plugs.Bev_masks{ii}.Dilated_boost_map;
  
  % generate tumor BEV
  [res] = ...
    maskbev_single_angle( handles.plugs.Bev_masks{ii}.Angle, ...
    handles.plugs.CTFrame.Tumor, handles.plugs.CTFrame.Tumor_CT, 1  );
  blah{ii}.Tumor = res{1}.Boost;
  
  [res] = ...
    maskbev_single_angle( handles.plugs.Bev_masks{ii}.Angle, ...
    handles.plugs.CTFrame.MRIoutline, handles.plugs.CTFrame.MRIoutline, 1);
  blah{ii}.Leg = res{1}.Boost;
  
  blah{ii}.AntiBoostProtection = handles.plugs.Bev_masks{ii}.Subtract_Dilated_boost_map;
  blah{ii}.AntiBoost = handles.plugs.Bev_masks{ii}.Antiboost_map;
end

fprintf('Variable tmp is created in the workspace.\n');
assignin('base', 'tmp', blah)
disp(blah);

% --------------------------------------------------------------------
function mExport3DMask_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function mViewContour_Callback(hObject, eventdata, handles)
% hh = [handles.mViewContour, handles.mViewBEV];
switch hObject
  case handles.mViewContour
    handles.drawing_method = 1;
    handles.mViewContour.Checked = 'on';
    handles.mViewBEV.Checked = 'off';
  case handles.mViewBEV
    handles.drawing_method = 2;
    handles.mViewContour.Checked = 'off';
    handles.mViewBEV.Checked = 'on';
end
guidata(handles.figure1, handles);
PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function pmBEVAngle_Callback(hObject, eventdata, handles)
PlotAll(hObject,handles.axes1, handles, handles.options);
nBev        = handles.pmBEVAngle.Value;
planeOffset = get(handles.slOffset, 'value');
set(handles.eLog, 'string', sprintf('nBev=%i\nOff=%4.2f',nBev,planeOffset));

% --------------------------------------------------------------------
function pbLoad_Callback(hObject, eventdata, handles)
layer = {};
layer{end+1}.Description = 'oxygen map';
layer{end}.Source = 'oxygen';
layer{end}.Method = 'asis';
layer{end}.Range = [0 40];
layer{end}.beam_alpha_miss = 0.5;
layer{end+1}.Description = 'radiaton beam';
layer{end}.Source = 'beamBEV';
layer{end}.Method = 'area';
layer{end}.color = [0.5 0.5 0.5];
layer{end+1}.Description = 'BEV leg';
layer{end}.Source = 'legBEV';
layer{end}.Method = 'contour';
layer{end}.color = 'b'; layer{end}.LW = 2;
layer{end+1}.Description = 'hypoxia contour';
layer{end}.Source = 'hypoxiaBEV';
layer{end}.Method = 'contour';
layer{end}.Threshold = 10;
layer{end}.color = 'r'; layer{end}.LW = 4;
% layer{end+1}.Description = 'hypoxia contour';
% layer{end}.Source = 'hypoxia';
% layer{end}.Method = 'contour';
% layer{end}.Threshold = 10;
% layer{end}.color = 'w';
layer{end+1}.Description = 'tumor contour';
layer{end}.Source = 'tumor';
layer{end}.Method = 'contour';
layer{end}.color = 'm'; layer{end}.LW = 4;

layer{end+1}.Description = 'global_settings';
layer{end}.Source = '';
layer{end}.Method = 'global';
layer{end}.range = [-10, 10, -10, 10];

setp.boost = true;
PlotAll2(handles, handles.axes1, setp, layer);

% PlotAll(hObject,handles.axes1, handles, handles.options);

% --------------------------------------------------------------------
function [ismaxedout,mask] = expand(mask, expansion_area, S)
ismaxedout = false;
se = strel('disk',3);
mask1 = mask;

while numel(find(mask)) < S
  mask1 = imdilate(mask1, se) & expansion_area;
  difference = find(mask1 & ~mask);
  if ~any(difference), ismaxedout = true; break; end
  if numel(find(mask1)) > S
    n1 = numel(find(mask1));
    mask1(difference(1:n1-S)) = false;
  end
  mask = mask1;
end

% --------------------------------------------------------------------
function mask = contract(mask, area, S)
se = strel('disk',3);
mask1 = mask;

while numel(find(mask)) > S
  mask1 = imerode(mask1, se) & area;
  difference = find(mask & ~mask1);
  if ~any(difference), break; end
  n1 = numel(find(mask1));
  if n1 > S
    fprintf('n1=%i S=%i\n', n1, S)
    mask1(difference) = false;
  else
    fprintf('n1=%i S=%i\n', n1, S)
    mask1(difference(1:S-n1)) = true;
  end
  mask = mask1;
end

% --------------------------------------------------------------------
function [is, mask1] = connect_islands(mask, block_area, CM)
is = true;
x = 1:size(block_area,1);
[X,Y]=meshgrid(x,x);
mask1 = false(size(mask));

% find islands
all = ~mask|block_area;
CC = bwconncomp(all);

for ii=1:length(CC.PixelIdxList)
  mm = false(size(block_area));
  mm(CC.PixelIdxList{ii}) = true;
  if mm(1,1), continue; end
  disp('Connecting islands.');
  CM = [sum(sum(mm.*X)), sum(sum(mm.*Y))]/sum(sum(mm));
  
  % expand island until it hits something
  all_but_one = all & ~mm;
  se1 = strel('disk', 3);
  mm1 = mm;
  while numel(find(all_but_one & mm1)) == 0
    mm1 = imdilate(mm1, se1);
  end
  % TODO
  % here we need to eliminate multiple overlaps
  
  % calculate CM for overlap
  overlap = all_but_one & mm1;
  cm = [sum(sum(overlap.*X)), sum(sum(overlap.*Y))]/sum(sum(overlap));
  
  % draw line from the center outwards
  dx = cm(1)-CM(1);
  dy = cm(2)-CM(2);
  if abs(dx) > abs(dy)
    dy_dx = (cm(2)-CM(2)) / (cm(1)-CM(1));
    for jj=min(CM(1)-cm(1),0):max(CM(1)-cm(1),0)
      mask1(floor(cm(2)+jj*dy_dx), floor(cm(1)+jj))=true;
    end
  else
    dx_dy = (cm(1)-CM(1)) / (cm(2)-CM(2));
    for jj=min(CM(2)-cm(2),0):max(CM(2)-cm(2),0)
      mask1(floor(cm(2)+jj), floor(cm(1)+dx_dy*jj))=true;
    end
  end
end
se = strel('disk',9);
mask1 = imdilate(mask1, se) & mask;

% --------------------------------------------------------------------
function CM = cm(X,Y,mask)
CM = [sum(sum(mask.*X))/sum(sum(mask)), sum(sum(mask.*Y))/sum(sum(mask))];

% --------------------------------------------------------------------
function mOtherAddAngle_Callback(hObject, eventdata, handles)
[answer] = inputdlg({'Enter new angle'},'Add angles',1,{'90:75:450'});
if ~isempty(answer)
  new_angles = eval(answer{1});
  for ii=1:length(new_angles)
    handles.plugs.Bev_masks{end+1}.Angle = new_angles(ii);
  end
  guidata(handles.figure1, handles);
  UpdateBeamSelectorList(hObject, handles, handles.plugs.Bev_masks);
end

% --------------------------------------------------------------------
function mOtherUseCache_Callback(hObject, ~, handles)
if isequal(handles.mOtherUseCache.Checked,'on')
  handles.mOtherUseCache.Checked = 'off';
else
  handles.mOtherUseCache.Checked = 'on';
end

% --------------------------------------------------------------------
function mViewPlots_Callback(hObject, eventdata, handles)
% hObject    handle to mViewPlots (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function mView2PanelPlot1_Callback(hObject, eventdata, handles)
figure(51); clf
set(gcf, 'renderer', 'painter')
% pst=epr_CalcAxesPos(2, 2, [0.05,0.05], [0.05,0.05]);
% for ii=1:2*2, hh(ii)=axes('Position', pst(ii,:)); end

hh(1:4) = gca;
lwcontour = 3;

f = @(x1, x2) struct('Source', x1, 'Method', x2);
layer = {};
layer{end+1} = f('oxygen', 'asis');
layer{end}.Range = [0 40];
layer{end}.beam_alpha_miss = 0.5;
layer{end+1} = f('beamBEV', 'contour');
layer{end}.color = 'k'; layer{end}.LW = lwcontour+2;
% layer{end+1} = f('legBEV', 'contour');
% layer{end}.color = 'b'; layer{end}.LW = 2;
layer{end+1} = f('hypoxiaBEV', 'contour');
layer{end}.Threshold = 10;
layer{end}.color = 'r'; layer{end}.LW = lwcontour+2;
% layer{end+1} = f('hypoxia', 'contour');
% layer{end}.Threshold = 10;
% layer{end}.color = 'w';
layer{end+1} = f( 'tumor', 'contour');
layer{end}.color = 'm'; layer{end}.LW = lwcontour;

layer{end+1} = f('', 'global');
layer{end}.range = [-10, 10, -10, 10];


setp.boost = true;
% PlotAll2(handles, hh(1), setp, layer);

setp.boost = false;
PlotAll2(handles, hh(2), setp, layer);

layer = {};
layer{end+1} = f('oxygen', 'none');
layer{end}.Range = [0 40];
layer{end}.beam_alpha_miss = 0.5;
layer{end+1} = f('beamBEV', 'area');
layer{end}.color = [0 0 0];
% layer{end+1} = f('legBEV', 'contour');
% layer{end}.color = 'b'; layer{end}.LW = 2;
% layer{end+1} = f('hypoxiaBEV', 'contour');
% layer{end}.Threshold = 10;
% layer{end}.color = 'r'; layer{end}.LW = lwcontour;
% layer{end+1} = f('hypoxia', 'contour');
% layer{end}.Threshold = 10;
% layer{end}.color = 'w';
% layer{end+1} = f( 'tumorBEV', 'contour');
% layer{end}.color = 'm'; layer{end}.LW = lwcontour;

layer{end+1} = f('', 'global');
layer{end}.range = [-10, 10, -10, 10];


setp.boost = true;
% PlotAll2(handles, hh(3), setp, layer);

setp.boost = false;
% PlotAll2(handles, hh(4), setp, layer);

function h = circle2(x,y,r)
d = r*2;
px = x-r;
py = y-r;
h = rectangle('Position',[px py d d],'Curvature',[1,1], 'LineStyle', ':');
daspect([1,1,1])


% --------------------------------------------------------------------
function mStackPlot_Callback(hObject, eventdata, handles)
options = [];
options.use_cross_lines = false;
options.use_hf10_contour = true;
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
options.isboost = true;
options.drawing_method = 1;
% options.apertures = 16:2:22;

h = figure;
pst = epr_CalcAxesPos(2, 5, [0.05,0.05], [0.05,0.05]);

for ii=1:10
  hax1 = axes('Position', pst(ii,:)); 
  options.planeoffset = (ii-5)*0.66;
  PlotAll(hObject, hax1, handles, options);
  title(hax1, sprintf("offset=%4.2f", options.planeoffset));
end
