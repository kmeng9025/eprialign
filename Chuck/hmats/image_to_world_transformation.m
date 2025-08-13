function mymat=image_to_world_transformation(im, scale);
% rotates, translates, scales coordinates in an image from pixel
% coordinates (i.e., [xpixel ypixel plane]) to world coordinates, i.e. [xcm,
% ycm, zcm].  Origin of world coordinates is at center of image volume, if
% the volume is 3d, otherwise at the center of slice number 1.
% 
% scaling happens last - note this is [sx sy sz]
%
% note this does NOT take into account matlab's (row, column) ordering.  if
% that's what is needed, use matrix_to_world_transformation instead.
%
%  C. Pelizzari Aug 2008
%

zoff = 1;
sz = size(im);
if (prod(size(sz)) > 2), zoff = (sz(3)+1) / 2; end
%
% even though we say we are not taking the row,col vs x,y ordering into
% account, still we are getting our sizes from a matlab matrix, and we need
% to handle the rows and cols properly.  presumably later we will be
% transforming (x,y,z) pixel coordinates, not (row, col, z), but still, the
% y direction will be along the columns.
% 
% 
xoff = (sz(2) + 1) / 2;
yoff = (sz(1) + 1) / 2;

orgmat=hmatrix_translate([-xoff -yoff -zoff]);


if (prod(size(scale)) > 1),
    myscale=[scale(1) scale(2) 1.0];
else
    myscale = [scale scale 1.0];
end
if (prod(size(scale)) > 2), myscale(3) = scale(3); end
scalemat=hmatrix_scale(myscale);

mymat=orgmat * scalemat;

