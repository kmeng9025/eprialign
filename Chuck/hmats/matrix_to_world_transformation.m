function mymat=matrix_to_world_transformation(im, scale);
% rotates, translates, scales coordinates in an image from matrix
% coordinates (i.e., [row column plane]) to world coordinates, i.e. [xcm,
% ycm, zcm].  Origin of world coordinates is at center of image volume, if
% the volume is 3d, otherwise at the center of slice number 1.
% 
% scaling happens last - note this is [sx sy sz]
%
% note the difference between this and image_to_world_transformation is
% this DOES take into account matlab's (row, column) vs (x,y) ordering.  
%
%  C. Pelizzari Aug 2008
%

zoff = 1;
sz = size(im);
if (prod(size(sz)) > 2), zoff = (sz(3)+1) / 2; end

xoff = (sz(2) + 1) / 2;
yoff = (sz(1) + 1) / 2;

orgmat=hmatrix_translate([-xoff -yoff -zoff]);

if (prod(size(scale)) > 1),
    myscale=[scale(1) -scale(2) 1.0];
else
    myscale = [scale -scale 1.0];
end
if (prod(size(scale)) > 2), myscale(3) = scale(3); end

scalemat=hmatrix_scale(myscale) * hmatrix_rotate_z(90) ;

mymat=orgmat * scalemat;

