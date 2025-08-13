% ARBUZ_SETIMAGE  set image property
% status = arbuz_SetImage(hGUI, set_list, field, value);
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

function status = arbuz_SetImage(hGUI, set_list, field, value)

% load the project
prj = getappdata(hGUI, 'project');
status = -1;

set_list = arbuz_ImageList(hGUI, set_list);

for ii=1:length(set_list)
  if set_list{ii}.SlaveIdx > 0
    im = prj.images{set_list{ii}.ImageIdx}.slaves{set_list{ii}.SlaveIdx};
  else
    im = prj.images{set_list{ii}.ImageIdx};
  end
  
  switch upper(field)
    case 'NAME'
      % modify transformations for master image
      if set_list{ii}.SlaveIdx == -1
        for jj=1:length(prj.Transformations)
          for kk=1:length(prj.Transformations{jj}.Matrices)
            if strcmp(prj.Transformations{jj}.Matrices{kk}.Image, im.Name)
              prj.Transformations{jj}.Matrices{kk}.Image = value;
            end
          end
        end
      end
      % change the name
      im.Name = value;
    case 'FILENAME'
      im.FileName = value;
    case 'DATA'
      im.data = value;
      im.isLoaded = true;
      prj.state = safeget(prj, 'state', 1) + 1;
    case 'DATA_INFO_MASK'
      im.data_info.Mask = value;
    case 'ISSTORE'
      im.isStore = value;
    case 'SELECTED'
      im.Selected = value;
    case 'VISIBLE'
      im.Visible = value;
    case 'BOX'
      im.box = value;
    case 'A'
      im.A = value;
      prj.state = safeget(prj, 'state', 1) + 1;
    case 'COLOR'
      if safeget(im, 'Selected', 0)
        im.SelectedColor = value;
      else
        im.NotSelectedColor = value;
      end
    case 'SELECTEDCOLOR'
      im.SelectedColor = value;
    case 'NOTSELECTEDCOLOR'
      im.NotSelectedColor = value;
    case 'LINK'
      im.Link = value;
    case 'APRE'
      if  set_list{ii}.SlaveIdx < 1, im.Apre = value; end
    case 'APRIME'
      if  set_list{ii}.SlaveIdx < 1, im.Aprime = value; end
    case 'ANEXT'
      if  set_list{ii}.SlaveIdx < 1, im.Anext = value; end
    case 'ASLAVE'
      if  set_list{ii}.SlaveIdx >= 1, im.A = value; end
    case 'ANATIVE'
      im.Anative = value;
    case 'TAG1'
      im.Tag1 = value;
    case 'TAG2'
      im.Tag2 = value;
    case 'TAG3'
      im.Tag3 = value;
    otherwise
      arbuz_ShowMessage(hGUI, ['arbuz_SetImage: Unknown field ''', field, '''.']);
  end
  
  if set_list{ii}.SlaveIdx > 0
    prj.images{set_list{ii}.ImageIdx}.slaves{set_list{ii}.SlaveIdx} = im;
  else
    prj.images{set_list{ii}.ImageIdx} = im;
  end
end

setappdata(hGUI, 'project', prj);


