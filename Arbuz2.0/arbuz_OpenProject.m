% ARBUZ_OPENPROJECT save projects to the specified file
% arbuz_SaveProject(hGUI, file_name);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function status = arbuz_OpenProject(hGUI, file_name)

status = 0;

set(hGUI, 'pointer', 'watch')
drawnow;

try
 % load_prj = load(file_name,'file_type');
 load_prj = load(file_name);
  
  if ~isfield(load_prj, 'file_type') || ...
    ~(strcmp(load_prj.file_type, 'Reg_v2.0') || strcmp(load_prj.file_type, 'CoReg_v1.0'))
    arbuz_ShowMessage(hGUI, sprintf('arbuz_OpenProject: File ''%s'' is not the Image registration project.', file_name));
    error(sprintf('arbuz_OpenProject: File ''%s'' is not the Image registration project.', file_name));
  end
catch err
  arbuz_ShowMessage(hGUI,err.message);
  error(err.message)
  return;
end
set(hGUI, 'pointer', 'arrow')

arbuz_InitializeProject(hGUI);
prj = getappdata(hGUI, 'project');
load_prj = load(file_name);

try
  switch load_prj.file_type
    case 'Reg_v2.0'
      prj.images = load_prj.images;

      prj.Transformations = safeget(load_prj, 'transformations', {});
      prj.Sequences = safeget(load_prj, 'sequences', {});
      prj.ActiveSequence = safeget(load_prj, 'activesequence', {});
      prj.ActiveTransformation = safeget(load_prj, 'activetransformation', {});
      prj.Groups = safeget(load_prj, 'groups', {});
      prj.saves = safeget(load_prj, 'saves', {});
      prj.comments = safeget(load_prj, 'comments', '');
      prj.state = safeget(load_prj, 'state', 1);
      prj.Status = safeget(load_prj, 'status', []);
    case 'CoReg_v1.0'
      for ii=1:length(load_prj.images)
         load_prj.images{ii}.slaves = load_prj.images{ii}.proxy;
         load_prj.images{ii} = rmfield(load_prj.images{ii},'proxy');
      end
      prj.images = load_prj.images;

      prj.Transformations = safeget(load_prj, 'transformations', {});
      prj.Sequences = safeget(load_prj, 'sequences', {});
      prj.ActiveSequence = safeget(load_prj, 'activesequence', {});
      prj.ActiveTransformation = safeget(load_prj, 'activetransformation', {});
      prj.Groups = safeget(load_prj, 'groups', {});
      prj.saves = safeget(load_prj, 'saves', {});
      prj.comments = safeget(load_prj, 'comments', '');
      prj.state = safeget(load_prj, 'state', 1);
      prj.Status = safeget(load_prj, 'status', []);
  end
   
  % create actions
  for ii=1:length(prj.Sequences)
    if ~isfield(prj.Sequences{ii}, 'Actions')
      prj.Sequences{ii}.Actions = {};
    end
  end
    
  arbuz_ShowMessage(hGUI, sprintf('arbuz_OpenProject: Project ''%s'' was opened.', file_name));
  prj.FileName = file_name;
  status = 1;

  % Verify all datasets
  for ii=1:length(prj.images)
    prj.images{ii}.isLoaded = safeget(prj.images{ii}, 'isLoaded', 1);
    for jj=1:length(prj.images{ii}.slaves)
      prj.images{ii}.slaves{jj}.isLoaded = safeget(prj.images{ii}.slaves{jj}, 'isLoaded', 1);
    end
  end

  % save the project
  setappdata(hGUI, 'project', prj);

  Status = arbuz_get(hGUI, 'Status');
  Keys = Status.keys;
  for ii=1:Status.Count
    if strcmpi(class(Status(Keys{ii})), 'double')
      fprintf('%s = %f\n', Keys{ii}, Status(Keys{ii}));
    elseif ischar(Status(Keys{ii}))
      fprintf('%s = %s\n', Keys{ii}, Status(Keys{ii}));
    end
  end
catch
  arbuz_ShowMessage(hGUI, sprintf('arbuz_OpenProject: ERROR. Project was not loaded.'));
end

clear load_prj;

% % load all required data
% for ii=1:length(prj.images)
%   if ~safeget(prj.images{ii}, 'isLoaded', 0) && ...
%       safeget(prj.images{ii}, 'Visible', 0)
%     [prj.images{ii}.data, prj.images{ii}.data_info] = ...
%       arbuz_LoadImage(prj.images{ii}.FileName, prj.images{ii}.ImageType);
%     prj.images{ii}.box = safeget(prj.images{ii}.data_info, ...
%       'Bbox', size(prj.images{ii}.data));
%     prj.images{ii}.Anative = safeget(prj.images{ii}.data_info, ...
%       'Anative', eye(4));
%     prj.images{ii}.isLoaded = 1;
%   end
% end
% 
% 
% 
% the_active_one = prj.Sequences{prj.ActiveSequence};
% str = {};
% for jj = 1:length(the_active_one.Sequence)
%   str{end+1} = ['Watch after stage: ', the_active_one.Sequence{jj}.Name];
% end
% set(prj.pmSeeStage, 'String', str, 'Value', the_active_one.WatchTransformation);
    


