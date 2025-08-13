function mymat=rotate_translate_scale_image_transform1(im,rotation, translation, scale, varargin);
% rotates, translates, scales coordinates in an image
% rotation is around image center
% translation follows rotation - note this is [tx ty]
% scaling happens last - note this is [sx sy]

% bbox is the bounding box of the region we want to transform:
% 2x2 array, row 1 is the lower left and row 2 the upper right corners
% [xll yll;
%  xur yur]
%
bbox = [0 0; size(im,2) size(im,1)];

if (nargin > 4) bbox = varargin{1}; end;
xsize = bbox(2,1)-bbox(1,1);
ysize = bbox(2,2)-bbox(1,2);

llcorner = bbox(1,:);

rotmat=hmatrix_rotate_z(rotation);
% pretrans takes center of bounding box to the origin
pretrans=inv(hmatrix_translate([llcorner(1)+xsize/2 llcorner(2)+ysize/2 0]));
% mytrans is the user supplied translation
mytrans=[translation(1) translation(2) 0.0];
% posttrans shifts center of bbox away from origin again
% so rotation is effectively about center of bbox
posttrans = hmatrix_translate([xsize/2 ysize/2 0]);
transmat=hmatrix_translate(mytrans);

if (prod(size(scale)) > 1),
    myscale=[scale(1) scale(2) 1.0];
else
    myscale = [scale scale 1.0];
end
scalemat=hmatrix_scale(myscale);

mymat=pretrans*rotmat*posttrans * transmat * scalemat;

