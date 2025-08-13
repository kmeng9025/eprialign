% 'previous', 'A', 'prime', 'next'
function status = arbuz_ApplyTransformation(hGUI, transformation_name, trans_dest)

% load the project
prj = getappdata(hGUI, 'project');
status = 0;

if nargin < 3, trans_dest = 'A'; end

switch upper(trans_dest)
  case 'PREVIOUS'
    for ii=1:length(prj.images)
      prj.images{ii}.Apre = prj.images{ii}.Apre * arbuz_GetTransformation(hGUI,transformation_name, prj.images{ii}.Name);
    end
  case 'A'
    for ii=1:length(prj.images)
      prj.images{ii}.A = prj.images{ii}.A * arbuz_GetTransformation(hGUI,transformation_name, prj.images{ii}.Name);
    end
  case 'PRIME'
    for ii=1:length(prj.images)
      prj.images{ii}.Aprime = prj.images{ii}.Aprime * arbuz_GetTransformation(hGUI,transformation_name, prj.images{ii}.Name);
    end
  case 'NEXT'
    for ii=1:length(prj.images)
      prj.images{ii}.Anext = prj.images{ii}.Anext * arbuz_GetTransformation(hGUI,transformation_name, prj.images{ii}.Name);
    end
  case 'FIX'
    for ii=1:length(prj.images)
      prj.images{ii}.A = prj.images{ii}.A * prj.images{ii}.Aprime;
      prj.images{ii}.Aprime = eye(4);
    end
  case 'RESET'
    for ii=1:length(prj.images)
      prj.images{ii}.Apre   = eye(4);
      prj.images{ii}.A      = eye(4);
      prj.images{ii}.Aprime = eye(4);
      prj.images{ii}.Anext  = eye(4);
    end
    status = 1;
end

setappdata(hGUI, 'project', prj);

