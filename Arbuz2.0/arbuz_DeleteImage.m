% ARBUZ_DELETEIMAGE  deletes an image from the project
% status = arbuz_DeleteImage(hGUI, image_list);
% hGUI - handle to the object that holds the project [double]
% image_name  - master image name or index (for dependent images only) [string or int]
% slave_image - slave image name or index (for dependent images only) [string or int]
% status - -1/0/1 for error/image deleted [int]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function  status = arbuz_DeleteImage(hGUI, image_list)

image_list = arbuz_ImageList(hGUI, image_list);

status = -1;

for ii=1:length(image_list)
  [midx, sidx, status] = arbuz_GetImageIdx(hGUI, image_list{ii});
  if midx < 0 || status == 0
    arbuz_ShowMessage(hGUI, 'arbuz_DeleteImage: Image was not deleted.');
    return;
  end
  prj = getappdata(hGUI, 'project');
  if sidx == -1
    Name = prj.images{midx}.Name;
    prj.images = prj.images([1:midx-1, midx+1:end]);
  else
    Name = [prj.images{midx}.Name,'/',prj.images{midx}.slaves{sidx}.Name];
    prj.images{midx}.slaves = prj.images{midx}.slaves([1:sidx-1, sidx+1:end]);
  end
  setappdata(hGUI, 'project', prj);
  arbuz_ShowMessage(hGUI, ['arbuz_DeleteImage: Image ''',Name,''' deleted.']);
end