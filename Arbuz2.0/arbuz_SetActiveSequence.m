function status = arbuz_SetActiveSequence(hGUI, active_seq)

% load the project
prj = getappdata(hGUI, 'project');
status = 1;

prj.ActiveSequence = active_seq;
arbuz_ShowMessage(hGUI, ['arbuz_SetActiveSequence: Sequence ''',...
  prj.Sequences{prj.ActiveSequence}.Name,''' was set as active.']);

setappdata(hGUI, 'project', prj);
