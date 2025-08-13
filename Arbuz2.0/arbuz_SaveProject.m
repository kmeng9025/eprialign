% ARBUZ_SAVEPROJECT save projects to the specified file
% arbuz_SaveProject(hGUI, file_name);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function status = arbuz_SaveProject(hGUI, file_name)

% load the project
prj = getappdata(hGUI, 'project');

status = 0;

try
  images    = prj.images;
  
  % unload not required data
  for ii=1:length(images)
    if ~safeget(images{ii}, 'isStore', 0)
      images{ii}.data = [];
      images{ii}.isLoaded = 0;
    end
  end
  
  
  for ii = 1: length(images)
  images{ii}.slaves = prj.images{ii}.slaves;
  end
  
  file_type = 'Reg_v2.0';
  transformations = prj.Transformations;
  sequences = prj.Sequences;
  groups = prj.Groups;
  activesequence = prj.ActiveSequence;
  activetransformation = prj.ActiveTransformation;
  saves = prj.saves;
  status = safeget(prj, 'Status', []);
  comments = safeget(prj, 'comments', '');
  
  save(file_name,'file_type', 'images','transformations', 'sequences', ...
    'activesequence','activetransformation', 'groups', 'saves', 'comments', 'status');
  
  arbuz_ShowMessage(hGUI, sprintf('arbuz_SaveProject: Project was saved to ''%s''.', file_name));
  status = 1;
  prj.can_close = 1;
  setappdata(hGUI, 'project', prj);
catch
  arbuz_ShowMessage(hGUI, sprintf('arbuz_SaveProject: ERROR. Project was not saved.'));
end