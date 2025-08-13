% ARBUZ_GETIMAGEIDX  get image index (both for master and dependent images)
% midx = arbuz_GetImageIdx(hGUI, image_alias);
% [midx, sidx, status] = arbuz_GetImageIdx(hGUI, image_alias);
% hGUI - handle to the object that holds the project [double]
% image_alias - see arbuz_ImageList(alias), alias parameter
% midx - index of master image [int]
% sidx - index of dependent image [int]
% status - 0/1 for error/OK [int]
% See also ARBUZ_IMAGELIST.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function [midx, sidx, status] = arbuz_GetImageIdx(hGUI, image_alias)

sidx = -1; % slave index
midx = -1; % master index
status = 0;

image_list = arbuz_ImageList(hGUI, image_alias);
if length(image_list) ~= 1
  arbuz_ShowMessage(hGUI, 'arbuz_GetImageIdx: None or too many images are specified.');
  return;
end

image_name = image_list{1};

% load the project
prj = getappdata(hGUI, 'project');

if isfield(image_name, 'ImageIdx'), midx = image_name.ImageIdx;
elseif isfield(image_name, 'Image')
  for ii=1:length(prj.images)
    if strcmp(prj.images{ii}.Name, image_name.Image)
      midx = ii;
      break;
    end
  end
  if midx == -1
    arbuz_ShowMessage(hGUI, ['arbuz_GetImageIdx: Image ''',image_name.Image,''' is not found.']);
    return;
  end
else
  arbuz_ShowMessage(hGUI, 'arbuz_GetImageIdx: Image description was not found.');
  return;
end
if isfield(image_name, 'SlaveIdx'), sidx = image_name.SlaveIdx; status = 1;
elseif isfield(image_name, 'Slave')
  for ii=1:length(prj.images{midx}.slaves)
    if strcmp(prj.images{midx}.slaves{ii}.Name, image_name.Slave)
      sidx = ii;
      break;
    end
  end
  if sidx == -1
    arbuz_ShowMessage(hGUI, ['arbuz_GetImageIdx: Dependent image ''',image_name.Slave,''' is not found.']);
    return;
  end
  status = 1;
else
  % this is master image
  status = 1;
end
