% function output_list = arbuz_GetSaveParList(hh, plugin_name)
% hh          - handle or handles structure of ArbuzGUI
% plugin_name - name of the plugin or empty for all

function output_list = arbuz_GetSaveParList(hGUI, plugin_name)

% load the project
prj = getappdata(hGUI, 'project');

output_list = {};

if ~isfield(prj, 'saves'), return; end

for ii=1:length(prj.saves)
  if iscell(prj.saves{ii}.name)
    if isempty(prj.saves{ii}.name)
      output_list{end+1} = '+';
    else
      output_list{end+1} = prj.saves{ii}.name{1};
    end
  else
    if isempty(plugin_name) || strcmp(prj.saves{ii}.plugin_name, plugin_name)
      output_list{end+1} = prj.saves{ii}.name;
    end
  end
end