function status = arbuz_AddSequence(hGUI, new_seq)

% load the project
prj = getappdata(hGUI, 'project');
status = 0;

for ii=1:length(prj.Sequences)
  if strcmp(prj.Sequences{ii}.Name, new_seq)
    arbuz_ShowMessage(hGUI, ['arbuz_AddSequence: Sequence ''',new_seq,''' already exists.']);
    return;
  end
end

seq.Name = new_seq;
seq.Sequence = {};
seq.ActiveTransformation = -1;
seq.WatchTransformation = -1;
seq.Actions = {};
prj.Sequences{end+1} = seq;
status = 1;

if prj.ActiveSequence == -1
  prj.ActiveSequence = 1;
end

setappdata(hGUI, 'project', prj);
