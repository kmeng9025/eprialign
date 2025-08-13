function [A, A1] = arbuz_util_transform_matrix(hGUI, image1, image2, pars)

find_source = arbuz_FindImage(hGUI, image1, ...
  '', '', {'Ashow', 'Name','ANative'});
image_dest = struct('Image',image2.Image);
find_dest = arbuz_FindImage(hGUI, image_dest, '', '', {'Ashow','Box','ANative'});
if isempty(find_source) || isempty(find_dest)
  return;
end

% A  = find_dest{1}.Ashow * inv(find_source{1}.Ashow);
% A1  = (inv(find_dest{1}.Anative)*find_dest{1}.Ashow) * inv(inv(find_source{1}.Anative)*find_source{1}.Ashow);

A  = find_dest{1}.Ashow \ find_source{1}.Ashow;
A1  = (find_dest{1}.Anative \ find_dest{1}.Ashow) / (find_source{1}.Anative \ find_source{1}.Ashow);
