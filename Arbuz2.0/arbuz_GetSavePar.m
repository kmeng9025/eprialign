function plugin_data = arbuz_GetSavePar(hGUI, plugin_data, the_save_name, plugin_name)

% load the project
prj = getappdata(hGUI, 'project');

if isfield(prj, 'saves')
  for ii=1:length(prj.saves)
    if strcmp(uncell(prj.saves{ii}.name), the_save_name)
      plugin_data = epr_combine_structs(plugin_data, prj.saves{ii}); 
      plugin_data.name = uncell(plugin_data);
      break;
    end
  end
end

function arg = uncell(arg)

if iscell(arg), arg = arg{1}; end