% ARBUZ_GETIMAGETRANSFORMATION recalculates image transformations
% status = arbuz_SetActiveTransformation(hGUI, ActiveSequence, ActiveTransformation);
% status = arbuz_SetActiveTransformation(hGUI, ActiveSequence);
% status = arbuz_SetActiveTransformation(hGUI);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function [Abefore, A, Aafter] = arbuz_GetImageTransformation(hGUI, the_image, ActiveSequence, ActiveTransformation, WatchTransformation)

% load the project
prj = getappdata(hGUI, 'project');

seq = prj.Sequences{ActiveSequence};

Abefore = eye(4);
A = eye(4);
Aafter = eye(4);
for ii=1:WatchTransformation
  AA =  arbuz_GetTransformation(hGUI,seq.Sequence{ii}.Name, the_image);
  
  if ii < ActiveTransformation
    Abefore = Abefore * AA;
  elseif ii == ActiveTransformation
    A = AA;
  else
    Aafter = Aafter * AA;
  end
end
