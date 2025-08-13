function status = arbuz_DeleteSequence(hGUI, delete_seq)

% load the project
prj = getappdata(hGUI, 'project');
status = 0;

if nargin == 1
  prj.Sequences = {};
  status = 1;
  arbuz_ShowMessage(hGUI, 'arbuz_DeleteSequence: Sequences were deleted.');
else
  for ii=1:length(prj.Sequences)
    if strcmp(prj.Sequences{ii}.Name, delete_seq)
      arbuz_ShowMessage(hGUI, ['arbuz_DeleteSequence: Sequence ''',delete_seq,''' is deleted.']);
      prj.Sequences = prj.Sequences([1:ii-1, ii+1:end]);
      status = 1;
      break;
    end
  end
end

if isempty(prj.Sequences)
  prj.ActiveSequence = -1;
else
  prj.ActiveSequence = min(prj.ActiveSequence, length(prj.Sequences));
end

if status == 1
  setappdata(hGUI, 'project', prj);
else
  arbuz_ShowMessage(hGUI, ['arbuz_DeleteSequence: Sequence ''',delete_seq,''' was not found.']);
end