% ARBUZ_STAGETOTRANSFORMATION copies image transformation matrices A & Aprime 
% into transformation
% status = arbuz_StageToTransformation(hGUI, the_transformation)
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function status = arbuz_StageToTransformation(hGUI, the_transformation)
status = 1;

% load the project
prj = getappdata(hGUI, 'project');

for ii=1:length(prj.images)
  AA = prj.images{ii}.A*prj.images{ii}.Aprime;
  arbuz_SetTransformation(hGUI, the_transformation, prj.images{ii}.Name, AA);
end
