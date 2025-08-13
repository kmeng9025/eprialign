% plugin_data = arbuz_SetupPar(hGUI, operation, the_save_name, plugin_data)
% function to store and load parameters for plugins and other engines
% operation = 'load', 'save', 'rename', 'delete'

function plugin_data = arbuz_SetupPar(hGUI, operation, the_save_name, plugin_data)

% load the project
prj = getappdata(hGUI, 'project');

switch operation
% --------------------------------------------------------------------
  case 'load'
    if isfield(prj, 'saves')
      for ii=1:length(prj.saves)
        if strcmp(uncell(prj.saves{ii}.name), the_save_name)
          plugin_data = epr_combine_structs(plugin_data, prj.saves{ii});
          plugin_data.name = uncell(plugin_data);
          arbuz_ShowMessage(hGUI, 'arbuz_SetupPar: Parameters are loaded.');
          break;
        end
      end
    end
% --------------------------------------------------------------------
  case 'save'
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
    arbuz_ShowMessage(hGUI, 'arbuz_SetupPar: Parameters are saved.');
% --------------------------------------------------------------------
  case 'rename'
    if isfield(prj, 'saves')
      for ii=1:length(prj.saves)
        if strcmp(uncell(prj.saves{ii}.name), the_save_name)
            prj.saves{ii}.name = plugin_data;
            setappdata(hGUI, 'project', prj);
            arbuz_ShowMessage(hGUI, 'arbuz_SetupPar: Parameters are renamed.');
          break;
        end
      end
    end
% --------------------------------------------------------------------
  case 'delete'
    if isfield(prj, 'saves')
      isFound = -1;
      for ii=1:length(prj.saves)
        if strcmp(uncell(prj.saves{ii}.name), the_save_name)
          isFound = ii;
          break;
        end
      end
      
      if isFound > 0
        prj.saves(isFound) = [];
        setappdata(hGUI, 'project', prj);
        arbuz_ShowMessage(hGUI, 'arbuz_SetupPar: Parameters are deleted.');
      end
      
    end
end

% --------------------------------------------------------------------

function arg = uncell(arg)

if iscell(arg), arg = arg{1}; end
