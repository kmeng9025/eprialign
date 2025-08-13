function [the_surface, Ascale] = arbuz_mask2surface(the_mask, scale_factor)

the_surface = [];
Ascale = eye(4);
if any(scale_factor ~= 1)
  new_sz   = fix(sz./sz_factor);
  old_size = new_sz.*sz_factor;
  shape_array = [sz_factor; new_sz]; shape_array = shape_array(:)';
  compressed_mask = reshape(im_mask(1:old_size(1),1:old_size(2),1:old_size(3)), shape_array);
  compressed_mask = squeeze(sum(sum(sum(compressed_mask, 1), 3), 5));
  [the_surface.face,the_surface.vert]=isosurface(compressed_mask, prod(sz_factor)/2);
  Ascale = hmatrix_translate(-[1 1 1]) * hmatrix_scale(sz_factor) * hmatrix_translate([1 1 1]+(sz_factor-[1 1 1])/2);
else
  [the_surface.face,the_surface.vert]=isosurface(the_mask, 0.5);
end
