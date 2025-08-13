function [imout, x, y, AAA] = project_image2D(im, hmat,varargin)
% this function serves for 2D visualization purposes
% projects image into plane defined by the transformation of image corners
% if no arguments are supplied:
%            the number of pixels is set so that it approximately match
%            number of pixels of original image

rows = 0; 
cols = 0;
nanval=0;

sz = size(im);
bbox(:,1) = [1;sz(2);sz(2);1];
bbox(:,2) = [1;1;sz(1);sz(1)];
bbox(:,3) = [0;0;0;0];

if (nargin > 2), nanval = varargin{1}; end;
if (nargin > 3), bbox = varargin{2}; end;
if (nargin > 4), rows = varargin{3}; end;
if (nargin > 5), cols = varargin{4}; end;

% calculate the transformed bounding box to give us our output image size
if (rows * cols == 0),
  bbox_prime = htransform_vectors(hmat,bbox);
  maxx = max(bbox_prime(:,1))-min(bbox_prime(:,1) + 1);
  maxy = max(bbox_prime(:,2))-min(bbox_prime(:,2) + 1);
  scale = max(sz)/max([maxx,maxy]);
  % create transformation that will bring this bbox to 
  % the origin of coordinates and scale to appropriate number of pixels
  Aadd = hmatrix_translate([-min(bbox_prime(:,1)), -min(bbox_prime(:,2)), 0]) * ...
    hmatrix_scale([scale scale 1])*hmatrix_translate([1,1,0]);
  
  bbox_prime2 = htransform_vectors(hmat*Aadd,bbox);
  
  xmax = max(bbox_prime2(:,1));
  ymax = max(bbox_prime2(:,2));
  rows = round(ymax);
  cols = round(xmax);
  
  x = linspace(min(bbox_prime(:,1)),max(bbox_prime(:,1)),cols);
  y = linspace(min(bbox_prime(:,2)),max(bbox_prime(:,2)),rows);
else
  Aadd = eye(4);
  xmax = cols;
  ymax = rows;
end

imclass = class(im);
nplanes = 1;
sz=size(im);
if (numel(sz) > 2), 
   nplanes = size(im, 3);
end
imout = zeros(rows, cols, nplanes,imclass);
hinv=inv(hmat*Aadd);
corners = [1 1 0; xmax 1 0; xmax ymax 0; 1 ymax 0];
tcorners=htransform_vectors(hinv,corners);

% now generate the coordinates of a (rows x cols) raster within the frame
% defined by the transformed corners.  x,y and z may all vary along each
% row and column in the general case.  This is just a lot of calls to
% linspace, but is encapsulated in generate_slice_coordinates.

[xp,yp]=generate_slice_coordinates(tcorners, cols, rows);

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
    if strcmp(imclass, 'uint8')
        myplane = cast(im, 'double');
        iplane=interp2(myplane, xp, yp,'linear',nanval);
        imout = cast(iplane, imclass);
    else
        imout = interp2(im, xp, yp,'linear',nanval);
    end
end

AAA = hmatrix_translate(-[x(1), y(1), 0])* ...
  hmatrix_scale([(length(x)-1)/(x(end)-x(1)), (length(y)-1)/(y(end)-y(1)), 1]) *...
  hmatrix_translate([1, 1, 0]);
