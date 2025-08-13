% Utility to extract pO2, MRI and CT related data from registration
% 
% returns structure res with images in CT frame:
%     Tumor_inCT       matrix
%     Hypoxia_inCT     matrix
%     MRIoutline_inCT  matrix
%     pO2_inCT         matrix or cell array of matrices
%
% options: 
%   Verbosity    1/0 show addtional information
%   isMRIOutline 1/0 load MRI outline
%   isLoadAllpO2 1/0 load all oxygen images as a cell array; the selected one
%                  will be marked bu nPO2

function res = load_IMRTdata_from_registration(hGUI, parameters)

Verbosity = safeget(parameters, 'Verbosity', 1);
isMRIOutline = safeget(parameters, 'MRIOutline', 0);
isLoadAllpO2 = safeget(parameters, 'LoadAllpO2', 0);

% locate CT image
if Verbosity, fprintf('load_IMRTdata_from_registration: Extracting CT ...\n'); end
image_CT = arbuz_FindImage(hGUI, 'master', 'ImageType', 'DICOM3D', {'slavelist'});
image_CT = arbuz_FindImage(hGUI, image_CT, 'InName', 'CT', {'AShow','slavelist'});
if isempty(image_CT), error('load_IMRTdata_from_registration: No CT image detected'); end

% locate pO2 image: image_PO2{nI}
if Verbosity, fprintf('load_IMRTdata_from_registration: Extracting pO2 ...\n'); end
image_PO2list = arbuz_FindImage(hGUI, 'master', 'InName', 'PO2', {});
% find image with 2 in it
for nI=1:length(image_PO2list)
  if ~isempty(strfind(image_PO2list{nI}.Image, '002')), break; end
  if ~isempty(strfind(image_PO2list{nI}.Image, '_2')), break; end
end
res.nPO2 = nI;

image_PO2 = arbuz_FindImage(hGUI, image_PO2list{nI}, '', '', {'data','Ashow','SlaveList'});

% transform tumor to CT coordinates
image_PO2_tumor = arbuz_FindImage(hGUI, image_PO2{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
transdata = arbuz_util_transform(hGUI, image_PO2_tumor, image_CT{1}, []);
res.Tumor_inCT = transdata.data;

% create hypoxia map in CT coordinates
Hypoxia = image_PO2{1}.data <= 10 & image_PO2{1}.data > -25;
transdata = arbuz_util_transform_data(hGUI, image_PO2{1}, double(Hypoxia), true, image_CT{1}, []);
res.Hypoxia_inCT = transdata.data;

% load all pO2 images
if isLoadAllpO2
  % transform tumor to CT coordinates
  image_PO2_tumor = arbuz_FindImage(hGUI, image_PO2{1}.SlaveList, 'Name', 'Tumor', {'data', 'AShow'});
  transdata = arbuz_util_transform(hGUI, image_PO2_tumor, image_CT{1}, []);
  res.Tumor_inCT = transdata.data;
  
  pO2list = {};
  for nI=1:length(image_PO2list)
    image_PO2 = arbuz_FindImage(hGUI, image_PO2list{nI}, '', '', {'data','Ashow','SlaveList'});
    
    % transform pO2 to CT coordinates
    transdata = arbuz_util_transform(hGUI, image_PO2{1}, image_CT{1}, []);
    res.pO2_inCT{nI} = transdata.data;
    pO2list{end+1} = image_PO2list{nI}.Image;
  end
  res.pO2list = pO2list;
  % Load only pO2 number 2
else
  image_PO2 = arbuz_FindImage(hGUI, image_PO2list{nI}, '', '', {'data','Ashow','SlaveList'});
  

% transform pO2 to CT coordinates
  transdata = arbuz_util_transform(hGUI, image_PO2{1}, image_CT{1}, []);
  res.pO2_inCT = transdata.data;
end

if Verbosity, fprintf('load_IMRTdata_from_registration: Extracting MRI ...\n'); end
image_MRI = arbuz_FindImage(hGUI, 'master', 'UPPERCASENAME', 'MRI_AXIAL', {'SLAVELIST'});
if isempty(image_MRI), error('load_IMRTdata_from_registration: No MRI image detected'); end

if Verbosity, fprintf('load_IMRTdata_from_registration: Producing MRI Outline ...\n'); end
if isMRIOutline
  outl_CT = arbuz_FindImage(hGUI, image_CT{1}.SlaveList, 'InName', 'Outline', {});
  outl_CT = arbuz_FindImage(hGUI, outl_CT, 'ImageType', '3DMASK', {'SLAVELIST', 'data'});
  if isempty(outl_CT)
    if Verbosity, fprintf('load_IMRTdata_from_registration:Transforming MRI outline to CT frame ...\n'); end
    outl = arbuz_FindImage(hGUI, 'master', 'UPPERCASENAME', 'MRI_AXIAL', {'SLAVELIST'});
    outls = arbuz_FindImage(hGUI, outl{1}.SlaveList, 'Name', 'Outline', {'data'});
    pars.dilate = 0;
    LEG = arbuz_util_transform(hGUI, outls{1}, image_CT{1}, pars);
    res.MRIoutline_inCT = LEG.data(1:300,1:300,1:300) > 0.5;
  else
    res.MRIoutline_inCT = outl_CT{1}.data > 0;
  end
end