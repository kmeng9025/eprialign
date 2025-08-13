function status = arbuz_DeleteTransformation(hGUI, delete_trans)

% load the project
prj = getappdata(hGUI, 'project');
status = 0;

if nargin == 1
  prj.Transformations = {};
  status = 1;
  arbuz_ShowMessage(hGUI, 'arbuz_DeleteTransformation: Transformations were deleted.');
else
  for ii=1:length(prj.Transformations)
    if strcmp(prj.Transformations{ii}.Name, delete_trans)
      arbuz_ShowMessage(hGUI, ['arbuz_DeleteTransformation: Transformation ''',delete_trans,''' is deleted.']);
      prj.Transformations = prj.Transformations([1:ii-1, ii+1:end]);
      status = 1;
      break;
    end
  end
end

% Validate sequence
for ii=1:length(prj.Sequences)
  for jj=1:length(prj.Sequences{ii}.Sequence)
    is_verified = false;
    for kk=1:length(prj.Transformations)
      if strcmp(prj.Sequences{ii}.Sequence{jj}.Name, prj.Transformations{kk}.Name)
        is_verified = true;
      end
    end
    if ~is_verified
      arbuz_ShowMessage(hGUI, ['arbuz_DeleteTransformation: Sequence stage ''',prj.Sequences{ii}.Sequence{jj}.Name,''' was unassigned.']);
      prj.Sequences{ii}.Sequence{jj}.Name = '';
    end
  end
end

if status == 1
  setappdata(hGUI, 'project', prj);
else
  arbuz_ShowMessage(hGUI, ['arbuz_DeleteTransformation: Transformation ''',delete_trans,''' was not found.']);
end