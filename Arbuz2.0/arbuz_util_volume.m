%function to determine volume of tumor
function V=arbuz_util_volume(the_image)

Vox = diag(the_image.Anative);
nVox = numel(find( the_image.data));
V=nVox*prod(Vox(1:3))*1E-3;