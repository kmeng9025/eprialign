% ARBUZ_SETTRANSFORMATION  set transformation
% A = arbuz_SetTransformation(hGUI, transformation_name, image_name, A);
% hGUI - handle to the object that holds the project [double]
% transformation_name  - transformation name [string]
% image_name  - image name [string]
% A - Transformation matrix [double 4x4]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function status = arbuz_SetTransformation(hGUI, transformation_name, image_name, A)

status = 0;

% load the project
prj = getappdata(hGUI, 'project');

for ii=1:length(prj.Transformations)
  if strcmp(prj.Transformations{ii}.Name, transformation_name)
    for jj=1:length(prj.Transformations{ii}.Matrices)
      if strcmp(prj.Transformations{ii}.Matrices{jj}.Image, image_name)
        prj.Transformations{ii}.Matrices{jj}.A = A;
        status = 1;
        break;
      end
    end
    if ~status
      prj.Transformations{ii}.Matrices{end+1}.Image = image_name;
      prj.Transformations{ii}.Matrices{end}.A = A;
      status = 1;
    end
    break;
  end
end

prj.can_close = 0;
setappdata(hGUI, 'project', prj);

if status
  arbuz_SetActiveTransformation(hGUI);
end

