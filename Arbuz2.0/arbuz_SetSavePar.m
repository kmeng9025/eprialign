function arbuz_SetSavePar(hGUI, the_save_name, plugin_name, plugin_data)

% load the project
prj = getappdata(hGUI, 'project');

plugin_data.plugin_name = plugin_name;
plugin_data.name = the_save_name;

if isfield(prj, 'saves')
  isFound = -1;
  for ii=1:length(prj.saves)
    if strcmp(prj.saves{ii}.name, the_save_name)
      isFound = ii; break;
    end
  end
  if isFound > 0
    prj.saves{isFound} = plugin_data;
  else
    prj.saves{end+1} = plugin_data;
  end
else
  prj.saves{1} = plugin_data;
end

setappdata(hGUI, 'project', prj);

arbuz_UpdateInterface(hGUI);
arbuz_ShowMessage(hGUI, 'arbuz_SetSavePar: Parameters are saved.');
