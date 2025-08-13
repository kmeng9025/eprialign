function varargout = RadiationPlanPLG(varargin)
% RADIATIONPLANPLG MATLAB code for RadiationPlanPLG.fig
%      RADIATIONPLANPLG, by itself, creates a new RADIATIONPLANPLG or raises the existing
%      singleton*.
%
%      H = RADIATIONPLANPLG returns the handle to a new RADIATIONPLANPLG or the handle to
%      the existing singleton*.
%
%      RADIATIONPLANPLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RADIATIONPLANPLG.M with the given input arguments.
%
%      RADIATIONPLANPLG('Property','Value',...) creates a new RADIATIONPLANPLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RadiationPlanPLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RadiationPlanPLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RadiationPlanPLG

% Last Modified by GUIDE v2.5 26-Mar-2014 16:28:22

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RadiationPlanPLG_OpeningFcn, ...
                   'gui_OutputFcn',  @RadiationPlanPLG_OutputFcn, ...
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
function RadiationPlanPLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for RadiationPlanPLG
handles.output = hObject;

% Add handle of calling object
handles.hh = varargin{1};

find_list3D = GetImageList(handles.hh);
 
handles.find_list = find_list3D;
str = cell(1, length(find_list3D));
for ii=1:length(find_list3D)
  str{ii}=find_list3D{ii}.FullName;
end
set(handles.pmOxygenImage, 'String', str);
set(handles.pmTargetImage, 'String', str);

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = RadiationPlanPLG_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;


% --------------------------------------------------------------------
function pmOxygenImage_Callback(hObject, eventdata, handles)
pos1 = get(handles.pmOxygenImage, 'Value');
fill_mask_menu(handles.pmTumorMask, handles, handles.find_list{pos1}, true);

% --------------------------------------------------------------------
function pmTumorMask_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pmTargetImage_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function eLog_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pbManual_Callback(hObject, eventdata, handles) %#ok<*INUSL,*DEFNU>
pos1 = get(handles.pmOxygenImage, 'Value');
% get source 3D masks 
ProxyList = arbuz_FindImage(handles.hh, {handles.find_list{pos1}}, ...
  '', '', {'SlaveList', 'data', 'Mask', 'Ashow'});

pos2 = get(handles.pmTumorMask, 'Value');
str = get(handles.pmTumorMask, 'String');

find_mask3D = arbuz_FindImage(handles.hh, ProxyList{1}.SlaveList, ...
    'Name', str{pos2}, {'Name', 'data'});

%RR = 3:8;
DD = [3 4 5 6.25 6.75 7.5 8 9]
RR = DD/(2*0.6629)
str = get(handles.eLog, 'String');
handles.target.set = {};

TumorMask = find_mask3D{1}.data&ProxyList{1}.Mask;
tic
for ii=1:length(RR)
    handles.target.set{ii} = [];
    [isocenter, res] = TargetHypoxicSphere(ProxyList{1}.data,RR(ii),TumorMask);
    handles.target.set{ii}.isocenter = isocenter;
    handles.target.set{ii}.result = res;
    
    [~,res2] = AvoidHypoxicSphericalShell(ProxyList{1}.data,RR(ii),TumorMask);
    
    str{end+1} = sprintf('C%i HF %4.2f(%i), NormF %4.2f(%i); CIO%i-%i HF %4.2f(%i) NormF %4.2f(%i) V%i', RR(ii), ...
        res.hypoxic_tumor_insphere_fraction, res.hypoxic_tumor_insphere_volume, ...
        res.normoxic_tumor_insphere_fraction, res.normoxic_tumor_insphere_volume, ...
        res2.inradius_1, res2.outradius_1, ...
        res2.hypoxic_tumor_inshell_fraction_1, res2.hypoxic_tumor_inshell_volume_1, ...
        res2.normoxic_tumor_inshell_fraction_1, res2.normoxic_tumor_inshell_volume_1, ...
        res2.tumor_inshell_1);    
    str{end+1} = sprintf('C%i HF %4.2f(%i), NormF %4.2f(%i); CIO%i-%i HF %4.2f(%i) NormF %4.2f(%i) V%i', RR(ii), ...
        res.hypoxic_tumor_insphere_fraction, res.hypoxic_tumor_insphere_volume, ...
        res.normoxic_tumor_insphere_fraction, res.normoxic_tumor_insphere_volume, ...
        res2.inradius_2, res2.outradius_2, ...
        res2.hypoxic_tumor_inshell_fraction_2, res2.hypoxic_tumor_inshell_volume_2, ...
        res2.normoxic_tumor_inshell_fraction_2, res2.normoxic_tumor_inshell_volume_2, ...
        res2.tumor_inshell_2);    
end
toc
str{end+1} = 'Done';
set(handles.eLog, 'String', str);

handles.target.A = ProxyList{1}.Ashow;

guidata(handles.figure1, handles);

% --------------------------------------------------------------------
function pmAlgorithm_Callback(hObject, eventdata, handles)

% RR = 3:8;
DD = [3 4 5 6.25 6.75 7.5 8 9]
RR = DD/(2*0.6629)
last_set = length(handles.target.set);

for ii = 1:length(handles.target.set)
    dHF(ii) = handles.target.set{ii}.result.hypoxic_tumor_insphere_fraction;
    dNF(ii) = handles.target.set{ii}.result.normoxic_tumor_insphere_fraction;
    hv(ii) = handles.target.set{ii}.result.hypoxic_tumor_insphere_volume;
    nv(ii) = handles.target.set{ii}.result.normoxic_tumor_insphere_volume;
    collimator(ii) = RR(ii)*4/3;
end

hvol = handles.target.set{last_set}.result.hypoxic_tumor_insphere_volume ./ handles.target.set{last_set}.result.hypoxic_tumor_insphere_fraction;
nvol = handles.target.set{last_set}.result.normoxic_tumor_insphere_volume ./ handles.target.set{last_set}.result.normoxic_tumor_insphere_fraction;
tvol = hvol+nvol;

figure(3); clf
subplot(2,1,1);
lw = 2;
h = plot(collimator, dHF, 'o-', collimator, dNF, '*-', ...
    collimator, (hv+nv)/tvol, 'linewidth', lw);
set(h(1),'MarkerSize', 12)
set(h(2),'MarkerSize', 12)
xlabel('Collimator diameter [mm]');
ylabel('Fraction of volume treated');
legend({'Hypoxic';'Normoxic';'Tumor'}, 2)
% title('Boost')

subplot(2,2,3); 
plot(dNF, dHF, 'ko-', 'linewidth', lw,'MarkerSize', 6); hold on
dr = repmat([0;1], 1, 6) - [dNF;dHF];
dd = sqrt(dr(1,:).^2 + dr(2,:).^2);
xlabel('FALSE positive');
ylabel('TRUE positive');
axis square
title('ROC');

subplot(2,2,4); 
plot(collimator, dd, 'ko-', 'linewidth', lw,'MarkerSize', 6)
xlabel('Collimator diameter [mm]');
ylabel('Optimization function')



% --------------------------------------------------------------------
function pbOk_Callback(hObject, eventdata, handles)
pos2 = get(handles.pmTargetImage, 'Value');
% get source 3D masks
ProxyList = arbuz_FindImage(handles.hh, {handles.find_list{pos2}}, ...
  '', '', {'Box', 'Ashow'});

user_choice = GetCollimator(handles);

idx = user_choice.set;
isocenter = handles.target.set{idx}.isocenter;

A = handles.target.A*inv(ProxyList{1}.Ashow);
Mask = false(ProxyList{1}.Box);
CP = round( htransform_vectors(A, isocenter([2,1,3])));
CP = CP([2,1,3]);

% get a boost  
RR =  round( htransform_vectors(A, user_choice.boost*[1,0,0;0,0,0;]));
RR = round(norm(diff(RR, 1))*[1,1,1]);

% get an Antiboost boost dimensions
URRin =  round( htransform_vectors(A, user_choice.antiboost_inner*[1,0,0;0,0,0;]));
URRin = round(norm(diff(URRin, 1))*[1,1,1]);
URRout =  round( htransform_vectors(A, user_choice.antiboost_outer*[1,0,0;0,0,0;]));
URRout = round(norm(diff(URRout, 1))*[1,1,1]);

% log the target
str = get(handles.eLog, 'String');
str{end+1} = sprintf('TX=%i TY=%i TZ=%i (R=%i)', CP(1), CP(2), CP(3), RR(1));
str{end+1} = 'Target mask is created.';
set(handles.eLog, 'String', str);

for pp = -1:1
    for ss = -1:1
        for ii=0:RR(1)*2
            Mask(fix(CP(1)+ii-RR(1)), CP(2)+pp, CP(3)+ss) = true;
        end
        for ii=0:RR(2)*2
            Mask(CP(1)+pp, fix(CP(2)+ii-RR(2)), CP(3)+ss) = true;
        end
        for ii=0:RR(3)*2
            Mask(CP(1)+pp, CP(2)+ss, fix(CP(3)+ii-RR(3))) = true;
        end
    end
end


% draw a cross over target
proxy_image.Name = 'Target';
proxy_image.isStore = 1;
proxy_image.data = Mask;
proxy_image.ImageType = '3DMASK';
arbuz_AddImage(handles.hh, proxy_image, ProxyList{1}.Image);

% draw a circle over target
proxy_image.Name = 'Boost';
proxy_image.isStore = 1;
proxy_image.data = epr_GetSphericMask(ProxyList{1}.Box, CP, mean(RR));
proxy_image.ImageType = '3DMASK';
arbuz_AddImage(handles.hh, proxy_image, ProxyList{1}.Image);

% draw a donut out of target
proxy_image.Name = 'Anti-Boost';
proxy_image.isStore = 1;
proxy_image.data = ~epr_GetSphericMask(ProxyList{1}.Box, CP, mean(URRin)) & ...
    epr_GetSphericMask(ProxyList{1}.Box, CP, mean(URRout));
proxy_image.handles.hh = '3DMASK';
arbuz_AddImage(handles.hh, proxy_image, ProxyList{1}.Image);

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pbCancel_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function pmCollimator_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function res = GetCollimator(handles)

switch get(handles.pmCollimator, 'Value')
    case 1, res.boost = 3; res.set = 1;
    case 2, res.boost = 4; res.set = 2;
    case 3, res.boost = 5; res.set = 3;
    case 4, res.boost = 6; res.set = 4;
    case 5, res.boost = 7; res.set = 5;
    case 6, res.boost = 8; res.set = 6;
    case 7, res.boost = 9; res.set = 7;
end

switch get(handles.pmABcollimatorIn, 'Value')
    case 1, res.antiboost_inner = 3;
    case 2, res.antiboost_inner = 4;
    case 3, res.antiboost_inner = 5;
    case 4, res.antiboost_inner = 6;
    case 5, res.antiboost_inner = 7;
    case 6, res.antiboost_inner = 8;
    case 7, res.antiboost_inner = 9;
end

switch get(handles.pmABcollimatorOut, 'Value')
    case 1, res.antiboost_outer = 9;
    case 2, res.antiboost_outer = 10;
    case 3, res.antiboost_outer = 11;
    case 4, res.antiboost_outer = 12;
    case 5, res.antiboost_outer = 13;
    case 6, res.antiboost_outer = 14;
    case 7, res.antiboost_outer = 15;
    case 8, res.antiboost_outer = 16;
    case 9, res.antiboost_outer = 17;
end
    

% --------------------------------------------------------------------
function fill_mask_menu(pm, handles, im_name, use_itself)
ProxyList = arbuz_FindImage(handles.hh, {im_name}, ...
  '', '', {'SlaveList'});
find_mask3D = arbuz_FindImage(handles.hh, ProxyList{1}.SlaveList, ...
  'ImageType', '3DMASK', {'Name'});

itself_pos = iff(use_itself, 1, 0);
str = cell(length(find_mask3D)+itself_pos, 1);
if use_itself, str{1} = 'itself'; end
for ii=1:length(find_mask3D), str{ii+itself_pos}=find_mask3D{ii}.Name; end
if isempty(str), str = {'None'}; end
  
set(pm, 'String', str, 'Value', 1)

% --------------------------------------------------------------------
function EvaluateRadiationPlan(handles)

radius = handles.target.radius;
isocenter = handles.target.isocenter;

% get source 3D image
pos1 = get(handles.pmOxygenImage, 'Value');
ProxyList = arbuz_FindImage(handles.hh, {handles.find_list{pos1}}, ...
  '', '', {'ProxyList', 'data', 'Mask'});

res = ManualTargetHypoxicSphere(ProxyList{1}.data,ProxyList{1}.Mask,radius,isocenter);

val = get(handles.pmCollimator, 'Value');
str = get(handles.pmCollimator, 'String');
msg1 = sprintf('X=%i Y=%i Z=%i (R=%i)', isocenter(1), isocenter(2), isocenter(3), radius);
msg2 = sprintf('%s: Hypoxic %5.2f (%i), normoxic %5.2f (%i)', str{val}, ...
    res.hypoxic_tumor_insphere_fraction, res.hypoxic_tumor_insphere_volume, ...
    res.normoxic_tumor_insphere_fraction, res.normoxic_tumor_insphere_volume);

str = get(handles.eLog, 'String');
str{end+1} = msg1;
str{end+1} = msg2;
set(handles.eLog, 'String', str);


% --------------------------------------------------------------------
function [results] = ManualTargetHypoxicSphere(po2image,Mask,radius,isocenter)
tumormask = Mask;

tumorvox = numel(find(tumormask == 1));
intumor = po2image(tumormask == 1);
hypoxvox = numel(find(intumor <= 10));
normoxvox = numel(find(intumor > 10));

[mask, distmat] = epr_GetSphericMask(Dim, isocenter, radius);

results.hypoxic_tumor_insphere_fraction = numel(find(po2image(po2image<=10 & tumormask==1 & distmat<=radius)))/hypoxvox;
results.hypoxic_tumor_insphere_volume = numel(find(po2image(po2image<=10 & tumormask==1 & distmat<=radius)));
results.normoxic_tumor_insphere_fraction  = numel(find(po2image(po2image>10 & tumormask==1 & distmat<=radius)))/normoxvox;
results.normoxic_tumor_insphere_volume  = numel(find(po2image(po2image>10 & tumormask==1 & distmat<=radius)));

% --------------------------------------------------------------------
function pbClearLog_Callback(hObject, eventdata, handles)
set(handles.eLog, 'String', {});

% --------------------------------------------------------------------
function pushbutton6_Callback(hObject, eventdata, handles)
find_list3D = GetImageList(handles.hh);
 
handles.find_list = find_list3D;
str = cell(1, length(find_list3D));
for ii=1:length(find_list3D)
  str{ii}=find_list3D{ii}.FullName;
end
set(handles.pmTargetImage, 'String', str);

% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function pushbutton7_Callback(hObject, eventdata, handles)
pos2 = get(handles.pmTargetImage, 'Value');
str2 = get(handles.pmTargetImage, 'String');
item_name = str2{pos2};
% get source 3D masks
ProxyList = arbuz_FindImage(handles.hh, handles.find_list(pos2), ...
  '', '', {'Box', 'Ashow'});

user_choice = GetCollimator(handles);

idx = user_choice.set;
isocenter = handles.target.set{idx}.isocenter;

A = handles.target.A*inv(ProxyList{1}.Ashow);
CP = round( htransform_vectors(A, isocenter([2,1,3])));
CP = CP([2,1,3]);

% get a boost dimensions
RR =  round( htransform_vectors(A, user_choice.boost*[1,0,0;0,0,0;]));
RR = round(norm(diff(RR, 1))*[1,1,1]);

% log the target
str = get(handles.eLog, 'String');
str{end+1} = sprintf('%s: TX=%i TY=%i TZ=%i (R=%i)', item_name, CP(1), CP(2), CP(3), RR(1));
set(handles.eLog, 'String', str);


function find_list3D = GetImageList(hh)

plist = {'FullName'};

find_list3D    = arbuz_FindImage(hh, 'master', 'ImageType', '3DEPRI', plist);
find_list3DPO2 = arbuz_FindImage(hh, 'master', 'ImageType', 'PO2_pEPRI', plist);
find_list3DAMP = arbuz_FindImage(hh, 'master', 'ImageType', 'AMP_pEPRI', plist);
find_list3DMRI = arbuz_FindImage(hh, 'master', 'ImageType', 'MRI', plist);
find_list3DAMIRA = arbuz_FindImage(hh, 'master', 'ImageType', 'AMIRA3D', plist);
find_list3DDICOM = arbuz_FindImage(hh, 'master', 'ImageType', 'DICOM3D', plist);

for ii=1:length(find_list3DPO2), find_list3D{end+1} = find_list3DPO2{ii}; end
for ii=1:length(find_list3DAMP), find_list3D{end+1} = find_list3DAMP{ii}; end
for ii=1:length(find_list3DMRI), find_list3D{end+1} = find_list3DMRI{ii}; end
for ii=1:length(find_list3DAMIRA), find_list3D{end+1} = find_list3DAMIRA{ii}; end
for ii=1:length(find_list3DDICOM), find_list3D{end+1} = find_list3DDICOM{ii}; end

