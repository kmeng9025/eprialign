function image2 = arbuz_util_transform_data(hGUI, image1, image1_data, ismask, image2, pars)

find_source = arbuz_FindImage(hGUI, image1, ...
  '', '', {'Ashow', 'Name','ANative'});
mask_name = image2.Slave;
image_dest = struct('Image',image2.Image);
find_dest = arbuz_FindImage(hGUI, image_dest, '', '', {'Ashow','Box'});
if isempty(find_source) || isempty(find_dest)
  return;
end

Dims = find_dest{1}.Box;

image2.Name = mask_name;
image2.isStore = 1;

cut_off = safeget(pars, 'cut_off', 0.5);
dilate_factor = safeget(pars, 'dilate_factor', 0);
algorithm = safeget(pars, 'algorithm', 'reslice');

if ~ismask
  image2.ImageType = find_source{1}.ImageType;
  image2.data = reslice_volume(inv(find_dest{1}.Ashow), inv(find_source{1}.Ashow), ...
    zeros(Dims), image1_data, 0, 1);
else
  % Transform mask
  image2.ImageType = '3DMASK';
  
  preprocessed_data = double(image1_data);
  
  if dilate_factor > 0
    res = diag(find_source{1}.Anative);
    se = epr_strel('ellipse', dilate_factor./res);
    %       se = epr_strel('sphere', dilate_factor);
    preprocessed_data = imdilate(preprocessed_data, se);
    preprocessed_data = double(preprocessed_data > 0.5);
  end
  
  switch algorithm
    case 'reslice' % reslicing
      A1 = find_dest{1}.Ashow;
      A2 = find_source{1}.Ashow;
      if isequal(A1,A2)
        image2.data = preprocessed_data;
      else
        image2.data = reslice_volume(inv(A1), inv(A2), ...
          zeros(Dims), preprocessed_data, 0, 1) > cut_off;
      end
    case 'voxel_project' % per-voxel projection
      sz = size(preprocessed_data);
      [Y,X,Z] = meshgrid(1:sz(2),1:sz(1),1:sz(3));
      
      hmat = find_source{ii}.Ashow*inv(find_dest{1}.Ashow);
      idx = preprocessed_data > 0.5;
      new_image.data = false(find_dest{1}.Box);
      v = htransform_vectors(hmat, [Y(idx),X(idx),Z(idx)]);
      for jj=1:3
        v(:,jj)=round(v(:,jj));
        for kk=1:size(v,1)
          v(kk,jj)=max([v(kk,jj), 1]);
          v(kk,jj)=min([v(kk,jj), find_source{1}.Box(jj)]);
        end
      end
      for jj=1:size(v,1)
        image2.data(v(jj,2),v(jj,1),v(jj,3)) = true;
      end
  end
end