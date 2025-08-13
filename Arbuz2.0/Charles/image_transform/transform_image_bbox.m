function imout = transform_image_bbox(im, hmat,varargin)
% produces a transformed (rotate, translate, scale - anything that can be
% represented in a 4x4 homogenous matrix) version of an image.  input image
% is "im" which may be any size and any number of components (e.g.,
% greyscale or RGB).  the matrix "hmat" takes a coordinate in the input
% image and maps it into the output image.  so to resample the input image
% into the space of the output, we actually take the inverse of hmat and
% transform the corners of the output image onto the input, then
% interpolate.
% size of output image is rows x cols.
% special care is taken if input image class is "uint8" since interpolation
% routines don't handle it properly.
% 
% varargin: first element is extrapolation value for pixels outside of the
% frame; second is user specified bounding box for region to be resampled %
% (i.e, cropping); third is user specified
% number of rows; fourth is user specified number of columns.
%
% C. Pelizzari Dec 2007
%
rows = 0; 
cols = 0;
nanval=0;

bbox = [0 0; size(im, 2) size(im,1)];

if (nargin > 2), nanval = varargin{1};, end;
if (nargin > 3), bbox = varargin{2};, end;
if (nargin > 4), rows = varargin{3}; end;
if (nargin > 5), cols = varargin{4}; end;

xsize = round(bbox(2,1)-bbox(1,1));
ysize = round(bbox(2,2)-bbox(1,2));

xmax = cols; 
ymax = rows;

if (rows * cols == 0),
% calculate the transformed bounding box to give us our output image size
% 
bbox3 = [bbox [0 0]']
tbox = htransform_vectors(hmat,bbox3);
xmax = max(tbox(:,1));
ymax = max(tbox(:,2));
rows = round(ymax);
cols = round(xmax);
end

imclass = class(im);
nplanes = 1;
sz=size(im);
if (prod(size(sz)) > 2), 
   nplanes = size(im, 3);
end
imout = zeros(rows, cols, nplanes,imclass);
hinv=inv(hmat);
corners = [1 1 0; xmax 1 0; xmax ymax 0; 1 ymax 0];
tcorners=htransform_vectors(hinv,corners);

% now generate the coordinates of a (rows x cols) raster within the frame
% defined by the transformed corners.  x,y and z may all vary along each
% row and column in the general case.  This is just a lot of calls to
% linspace, but is encapsulated in generate_slice_coordinates.

[xp,yp,zout]=generate_slice_coordinates(tcorners, cols, rows);

if (nplanes > 1),

    for n = 1:nplanes
        myplane = squeeze(im(:,:,n));
        if (imclass == 'uint8'), 
            myplane = cast(myplane, 'double');
            imout(:,:,n) = cast(interp2(myplane, xp, yp,'linear',nanval), imclass);
        else
            imout(:,:,n) = interp2(myplane, xp, yp,'linear',nanval);
        end
    end
else
    if (imclass == 'uint8'), 
        myplane = cast(im, 'double');
        iplane=interp2(myplane, xp, yp,'linear',nanval);
        size(iplane)
        imout = cast(iplane, imclass);
    else
        imout = interp2(im, xp, yp,'linear',nanval);
    end
end

