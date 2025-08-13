% ARBUZ_GETTRANSFORMATION  get transformation
% A = arbuz_GetTransformation(hGUI, transformation_name, image_name);
% hGUI - handle to the object that holds the project [double]
% transformation_name  - transformation name [string]
% image_name  - image name [string]
% A - Transformation matrix [double 4x4]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function A = arbuz_GetTransformation(hGUI, transformation_name, image_name)

% empty transformation
A = eye(4);

% load the project
prj = getappdata(hGUI, 'project');

for ii=1:length(prj.Transformations)
  if strcmp(prj.Transformations{ii}.Name, transformation_name)
    T = prj.Transformations{ii};
    for jj=1:length(T.Matrices)
      if strcmp(T.Matrices{jj}.Image, image_name)
        A = T.Matrices{jj}.A;
        break;
      end
    end
    break;
  end
end

