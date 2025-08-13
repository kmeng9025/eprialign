% ARBUZ_ADDIMAGE  adds an image to the project
% status = arbuz_AddImage(hGUI, new_image);
% status = arbuz_AddImage(hGUI, new_image, master_image);
% hGUI - handle to the object that holds the project [double]
% new_image - [structure] of new image parameters
%     [].Name - Image name [string]
%     [].ImageType  - Image type [string, 2D/3DEPR/CONTOUR/XYZ/3DSURFACE/
%         PO2_pEPRI/AMP_pEPRI/MRI/3DMASK/RAW/AMIRA3D/DICOM3D/SHAPE3D/GENERIC]
%     [].Anative  - Transformation from pixel/voxel to space [double, 4x4]
%     [].FileName - File name, [string]
%     [].isLoaded - Data status 0/1 - always loaded/loaded on demand [int, Reconstruction code [string, C/MATLAB/FORTRAN]
%     [].slaves   - cell array structure of slave images of the type new_image [cell array]
% master_image - master image name or index (for dependent images only) [string or int]
% status - -1/0/1 for error/image substituted/i,age replaced [int]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function  status = arbuz_AddImage(hGUI, new_image, master_image)

% load the project
prj = getappdata(hGUI, 'project');
status = -1;

Name = safeget(new_image, 'Name', '');
if isempty(Name)
  arbuz_ShowMessage(hGUI, 'arbuz_AddImage: Image does not have name.');
  return;
end

new_image.ImageType = safeget(new_image, 'ImageType', '');
new_image.Anative   = safeget(new_image, 'Anative', eye(4));
new_image.FileName  = safeget(new_image, 'FileName', '');
new_image.isLoaded  = safeget(new_image, 'isLoaded', 0);

box = [0,0,0];
switch new_image.ImageType
  case '2D', sz = size(new_image.data); box = [sz(1), sz(2), 0];
end
new_image.box     = safeget(new_image, 'box', box);

if nargin == 2
  new_image.slaves     = safeget(new_image, 'slaves', {});
  master_image_idx = arbuz_GetImageIdx(hGUI, {Name});
  if master_image_idx <= 0
    new_image.Selected  = safeget(new_image, 'Selected', 0);
    new_image.Visible   = safeget(new_image, 'Visible', 0);
    prj.images{end+1} = new_image; status = 1;
    arbuz_ShowMessage(hGUI, ['arbuz_AddImage: Image ''',Name,''' is added.']);
  else
    new_image.Selected  = safeget(prj.images{master_image_idx}, 'Selected', 0);
    new_image.Visible   = safeget(prj.images{master_image_idx}, 'Visible', 0);
%     new_image.Color   = safeget(prj.images{master_image_idx}, 'Color', 0);
    prj.images{master_image_idx} = new_image; status = 0;
    arbuz_ShowMessage(hGUI, ['arbuz_AddImage: Image ''',Name,''' is substituted.'])
  end
else
  [master_image_idx, slave_image_idx] = arbuz_GetImageIdx(hGUI, {master_image, Name});
  if master_image_idx == -1
    arbuz_ShowMessage(hGUI, 'arbuz_AddImage: Master image was not found.')
    return;
  else
    
  if slave_image_idx <= 0
    new_image.Selected  = safeget(new_image, 'Selected', 0);
    new_image.Visible   = safeget(new_image, 'Visible', 0);
    prj.images{master_image_idx}.slaves{end+1} = new_image; status = 1;
    arbuz_ShowMessage(hGUI, ['arbuz_AddImage: Dependent image ''',Name,''' is added.']);
    new_image.A         = eye(4);
  else
    new_image.Selected  = safeget(prj.images{master_image_idx}.slaves{slave_image_idx}, 'Selected', 0);
    new_image.Visible   = safeget(prj.images{master_image_idx}.slaves{slave_image_idx}, 'Visible', 0);
    if ~isfield(new_image, 'SelectedColor') && isfield(prj.images{master_image_idx}.slaves{slave_image_idx}, 'SelectedColor')
      new_image.SelectedColor   = prj.images{master_image_idx}.slaves{slave_image_idx}.SelectedColor;
    end
    if ~isfield(new_image, 'NotSelectedColor') && isfield(prj.images{master_image_idx}.slaves{slave_image_idx}, 'NotSelectedColor')
      new_image.NotSelectedColor   = prj.images{master_image_idx}.slaves{slave_image_idx}.NotSelectedColor;
    end
    prj.images{master_image_idx}.slaves{slave_image_idx} = new_image; status = 0;
    arbuz_ShowMessage(hGUI, ['arbuz_AddImage: Dependent image ''',Name,''' is substituted.'])
  end
  end
end

prj.can_close = 0;
prj.state = safeget(prj, 'state', 1) + 1;
setappdata(hGUI, 'project', prj);