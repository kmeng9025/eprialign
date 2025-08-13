function status = arbuz_AddGroup(hGUI, group_name, group_data)

% load the project
prj = getappdata(hGUI, 'project');
status = 0;

%   strip indexes
for ii=1:length(group_data)
  if isfield(group_data{ii}, 'ImageIdx')
    group_data{ii}=rmfield(group_data{ii}, 'ImageIdx');
  end
  if isfield(group_data{ii}, 'SlaveIdx')
    group_data{ii}=rmfield(group_data{ii}, 'SlaveIdx');
  end
end
  
for ii=1:length(prj.Groups)
  if strcmp(prj.Groups{ii}.Name, group_name)
    arbuz_ShowMessage(hGUI, ['arbuz_AddGroup: Group ''',group_name,''' was updated.']);
    prj.Groups{ii}.list = group_data;
    setappdata(hGUI, 'project', prj);
    return;
  end
end

the_group.Name = group_name;
the_group.list = group_data;
prj.Groups{end+1} = the_group;
status = 1;
arbuz_ShowMessage(hGUI, ['arbuz_AddGroup: Group ''',group_name,''' was added.']);
setappdata(hGUI, 'project', prj);
