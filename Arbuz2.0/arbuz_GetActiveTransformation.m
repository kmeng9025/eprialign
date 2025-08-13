function status = arbuz_GetActiveTransformation(hGUI)

prj = getappdata(hGUI, 'project');
idx = prj.Sequences{prj.ActiveSequence}.ActiveTransformation;

status = prj.Transformations{idx}.Name;