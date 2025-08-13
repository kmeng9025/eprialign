function varargout = RadiationPlan2PLG(varargin)
% RADIATIONPLAN2PLG MATLAB code for RadiationPlan2PLG.fig
%      RADIATIONPLAN2PLG, by itself, creates a new RADIATIONPLAN2PLG or raises the existing
%      singleton*.
%
%      H = RADIATIONPLAN2PLG returns the handle to a new RADIATIONPLAN2PLG or the handle to
%      the existing singleton*.
%
%      RADIATIONPLAN2PLG('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in RADIATIONPLAN2PLG.M with the given input arguments.
%
%      RADIATIONPLAN2PLG('Property','Value',...) creates a new RADIATIONPLAN2PLG or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before RadiationPlan2PLG_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to RadiationPlan2PLG_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help RadiationPlan2PLG

% Last Modified by GUIDE v2.5 18-Nov-2015 10:20:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @RadiationPlan2PLG_OpeningFcn, ...
                   'gui_OutputFcn',  @RadiationPlan2PLG_OutputFcn, ...
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
function RadiationPlan2PLG_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for RadiationPlan2PLG
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

% Set collimator diameters here
handles.DBOOST = [3 4 5 6.25 6.75 7.5 8 9];
handles.DISHELL = [3 4 5 6.25 6.75 7.5 8 9];

for ii = 1:length(handles.DISHELL)
handles.DOSHELL(ii) = handles.DISHELL(ii)*2^(1/3)
end

str = {};
for ii=1:length(handles.DBOOST)
  str{ii} = sprintf('Cd=%4.2fmm',handles.DBOOST(ii));
end
set(handles.pmCollimator, 'string', str)

str = {};
for ii=1:length(handles.DISHELL)
  str{ii} = sprintf('Cid=%4.2fmm',handles.DISHELL(ii));
end
set(handles.pmABcollimatorIn, 'string', str)

str = {};
for ii=1:length(handles.DOSHELL)
  str{ii} = sprintf('Cod=%4.2fmm',handles.DOSHELL(ii));
end
set(handles.pmABcollimatorOut, 'string', str)


% Update handles structure
guidata(hObject, handles);

% --------------------------------------------------------------------
function varargout = RadiationPlan2PLG_OutputFcn(hObject, eventdata, handles) 
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

str = get(handles.eLog, 'String');
handles.target.set = {};

TumorMask = find_mask3D{1}.data&ProxyList{1}.Mask;
tic
for ii=1:length(handles.DBOOST)
    handles.target.set{ii} = [];
%     [isocenter, res] = TargetHypoxicSphere(ProxyList{1}.data,RR(ii),TumorMask);
    [isocenter, res] = TargetHypoxicSphere2(ProxyList{1}.data,[0.6629,0.6629,0.6629],TumorMask, handles.DBOOST(ii)/2);
    handles.target.set{ii}.isocenter = isocenter;
    handles.target.set{ii}.result = res;
    
%     [~,res2] = AvoidHypoxicSphericalShell(ProxyList{1}.data,RR(ii),TumorMask);
    [~,res2] = AvoidHypoxicSphericalShell2(ProxyList{1}.data,[0.6629,0.6629,0.6629],TumorMask, handles.DBOOST(ii)/2, res);
    
    str{end+1} = sprintf('C%4.2f HF %4.2f(%4.1f), NormF %4.2f(%4.1f); CIO%i-%i HF %4.2f(%i) NormF %4.2f(%i) V%i', handles.DBOOST(ii), ...
        res.hypoxic_tumor_insphere_fraction, res.hypoxic_tumor_insphere_volume, ...
        res.normoxic_tumor_insphere_fraction, res.normoxic_tumor_insphere_volume, ...
        res2.inradius_1, res2.outradius_1, ...
        res2.hypoxic_tumor_inshell_fraction_1, res2.hypoxic_tumor_inshell_volume_1, ...
        res2.normoxic_tumor_inshell_fraction_1, res2.normoxic_tumor_inshell_volume_1, ...
        res2.tumor_inshell_1);    
    str{end+1} = sprintf('C%4.2f HF %4.2f(%4.1f), NormF %4.2f(%4.1f); CIO%i-%i HF %4.2f(%i) NormF %4.2f(%i) V%i', handles.DBOOST(ii), ...
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

last_set = length(handles.target.set);

for ii = 1:length(handles.target.set)
    dHF(ii) = handles.target.set{ii}.result.hypoxic_tumor_insphere_fraction;
    dNF(ii) = handles.target.set{ii}.result.normoxic_tumor_insphere_fraction;
    hv(ii) = handles.target.set{ii}.result.hypoxic_tumor_insphere_volume;
    nv(ii) = handles.target.set{ii}.result.normoxic_tumor_insphere_volume;
    collimator(ii) = handles.DBOOST(ii);
end

hvol = handles.target.set{last_set}.result.hypoxic_tumor_insphere_volume ./ handles.target.set{last_set}.result.hypoxic_tumor_insphere_fraction;
nvol = handles.target.set{last_set}.result.normoxic_tumor_insphere_volume ./ handles.target.set{last_set}.result.normoxic_tumor_insphere_fraction;
tvol = hvol+nvol;

figure(3); clf
subplot(2,1,1);
lw = 2;
h = plot(collimator, dHF, 'o-', collimator, dNF, '*-', ...
    collimator, (hv+nv)/tvol, 'x-', collimator, 0.5*ones(size(collimator)), ':',...
    'linewidth', lw);
set(h(1),'MarkerSize', 12)
set(h(2),'MarkerSize', 12)
xlabel('Collimator diameter [mm]');
ylabel('Fraction of volume treated');
legend({'Hypoxic';'Normoxic';'Tumor'}, 'location', 'northwest')
% title('Boost')

subplot(2,2,3); 
plot(dNF, dHF, 'ko-', 'linewidth', lw,'MarkerSize', 6); hold on
dr = repmat([0;1], 1, length(dNF)) - [dNF;dHF];
dd = sqrt(dr(1,:).^2 + dr(2,:).^2);
xlabel('FALSE positive');
ylabel('TRUE positive');
axis([0,1,0,1])
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
DD =  round( htransform_vectors(inv(ProxyList{1}.Ashow), user_choice.boost*[1,0,0;0,0,0;]));
DD = round(norm(diff(DD, 1))*[1,1,1]);

% get an Antiboost boost dimensions
URRin =  round( htransform_vectors(inv(ProxyList{1}.Ashow), user_choice.antiboost_inner*[1,0,0;0,0,0;]));
URRin = round(norm(diff(URRin, 1))*[1,1,1]);
URRout =  round( htransform_vectors(inv(ProxyList{1}.Ashow), user_choice.antiboost_outer*[1,0,0;0,0,0;]));
URRout = round(norm(diff(URRout, 1))*[1,1,1]);

% log the target
str = get(handles.eLog, 'String');
str{end+1} = sprintf('TX=%i TY=%i TZ=%i (Dvox=%i)', CP(1), CP(2), CP(3), DD(1));
str{end+1} = 'Target mask is created.';
set(handles.eLog, 'String', str);

for pp = -1:1
    for ss = -1:1
        for ii=0:DD(1)
            Mask(fix(CP(1)+ii-DD(1)/2), CP(2)+pp, CP(3)+ss) = true;
        end
        for ii=0:DD(2)
            Mask(CP(1)+pp, fix(CP(2)+ii-DD(2)/2), CP(3)+ss) = true;
        end
        for ii=0:DD(3)
            Mask(CP(1)+pp, CP(2)+ss, fix(CP(3)+ii-DD(3)/2)) = true;
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
proxy_image.data = epr_GetSphericMask(ProxyList{1}.Box, CP, mean(DD)/2);
% proxy_image.data(CP(1),CP(2),CP(3)+[-10:10]) = false;
% proxy_image.data(CP(1)+[-10:10],CP(2),CP(3)) = false;
% proxy_image.data(CP(1),CP(2)+[-10:10],CP(3)) = false;
proxy_image.ImageType = '3DMASK';
arbuz_AddImage(handles.hh, proxy_image, ProxyList{1}.Image);

% draw a donut out of target
proxy_image.Name = 'Anti-Boost';
proxy_image.isStore = 1;
proxy_image.data = ~epr_GetSphericMask(ProxyList{1}.Box, CP, mean(URRin)/2) & ...
    epr_GetSphericMask(ProxyList{1}.Box, CP, mean(URRout)/2);
proxy_image.handles.hh = '3DMASK';
arbuz_AddImage(handles.hh, proxy_image, ProxyList{1}.Image);

arbuz_UpdateInterface(handles.hh);

% --------------------------------------------------------------------
function pmCollimator_Callback(hObject, eventdata, handles)

% --------------------------------------------------------------------
function res = GetCollimator(handles)

res.set   = get(handles.pmCollimator, 'Value');
res.boost = handles.DBOOST(res.set); 

idx   = get(handles.pmABcollimatorIn, 'Value');
res.antiboost_inner = handles.DISHELL(idx); 

idx   = get(handles.pmABcollimatorOut, 'Value');
res.antiboost_outer = handles.DOSHELL(idx); 
    

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
function pbClearLog_Callback(hObject, eventdata, handles)
set(handles.eLog, 'String', {});

% --------------------------------------------------------------------
function pbUpdate_Callback(hObject, eventdata, handles)
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
function pbShowXYZ_Callback(hObject, eventdata, handles)
pos2 = get(handles.pmTargetImage, 'Value');
str2 = get(handles.pmTargetImage, 'String');
item_name = str2{pos2};
% get source 3D masks
ProxyList = arbuz_FindImage(handles.hh, handles.find_list(pos2), ...
  '', '', {'Box', 'Ashow','Name'});

user_choice = GetCollimator(handles);

idx = user_choice.set;
isocenter = handles.target.set{idx}.isocenter;

A = handles.target.A*inv(ProxyList{1}.Ashow);
CP = round( htransform_vectors(A, isocenter([2,1,3])));
CP = CP([2,1,3]);

% get a boost dimensions
DD =  round( htransform_vectors(inv(ProxyList{1}.Ashow), user_choice.boost*[1,0,0;0,0,0;]));
DD = round(norm(diff(DD, 1))*[1,1,1]);

% log the target
str = get(handles.eLog, 'String');
str{end+1} = sprintf('%s: TX=%i TY=%i TZ=%i (%s:Dvox=%i)', item_name, CP(1), CP(2), CP(3), ProxyList{1}.Name, DD(1));
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
