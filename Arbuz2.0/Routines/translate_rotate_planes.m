function newpoints = translate_rotate_planes(pointlist, angles, translations, center)

nplanes = size(pointlist, 1);

hmats = zeros(4,4,nplanes);

for i = 1: nplanes
    hmats(:,:,i) = rotate_about_center(hmatrix_rotate_z(angles(i)), center)*hmatrix_translate(translations(i,:));
end

newpoints = htransform_planes(pointlist, hmats);