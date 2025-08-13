function status = arbuz_AddTransformation(hGUI, new_trans)

% load the project
prj = getappdata(hGUI, 'project');
status = 0;

for ii=1:length(prj.Transformations)
  if strcmp(prj.Transformations{ii}.Name, new_trans)
    arbuz_ShowMessage(hGUI, ['arbuz_AddTransformation: Transformation ''',new_trans,''' already exists.']);
    return;
  end
end

trans.Name = new_trans;
trans.Matrices = {};
prj.Transformations{end+1} = trans;
status = 1;
arbuz_ShowMessage(hGUI, ['arbuz_AddTransformation: Transformation ''',new_trans,''' was added.']);

setappdata(hGUI, 'project', prj);
