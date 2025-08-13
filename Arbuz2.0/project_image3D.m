function [imout, x, y, AAA] = project_image3D(im, hmat, varargin)
% this function serves for 2D visualization purposes
% projects image into plane defined by the transformation of image corners
% if no arguments are supplied:
%            the number of pixels is set so that it approximately match
%            number of pixels of original image

rows = 0; 
cols = 0;
nanval=0;

sz = size(im);
im_bbox(:,1) = [1;sz(2);sz(2);1; 1;sz(2);sz(2);1];
im_bbox(:,2) = [1;1;sz(1);sz(1); 1;1;sz(1);sz(1)];
im_bbox(:,3) = [1;1;1;1;         sz(3);sz(3);sz(3);sz(3)];
bbox_prime = htransform_vectors(hmat, im_bbox);

if (nargin > 2), bbox = varargin{1}; end;
if (nargin > 4), rows = varargin{3}; end;
if (nargin > 5), cols = varargin{4}; end;

% calculate the transformed bounding box to give us our output image size
if (rows * cols == 0),
  maxx = max(bbox_prime(:,1))-min(bbox_prime(:,1) + 1);
  maxy = max(bbox_prime(:,2))-min(bbox_prime(:,2) + 1);
  scale = max(sz)/max([maxx,maxy]);
  % create transformation that will bring this bbox to 
  % the origin of coordinates and scale to appropriate number of pixels
  Aadd = hmatrix_translate([-min(bbox_prime(:,1)), -min(bbox_prime(:,2)), 0]) * ...
    hmatrix_scale([scale scale 1])*hmatrix_translate([1,1,0]);
  bbox_prime2 = htransform_vectors(hmat*Aadd, im_bbox);

  xmax = max(bbox_prime2(:,1));
  ymax = max(bbox_prime2(:,2));
  rows = round(ymax);
  cols = round(xmax);
else
  Aadd = eye(4);
  xmax = cols;
  ymax = rows;
end

if isempty(bbox)
  bbox = [min(bbox_prime(:,1)), max(bbox_prime(:,1)), ...
    min(bbox_prime(:,2)),max(bbox_prime(:,2))];
end

interpolate_bbox(:,1) = [bbox(1);bbox(2);bbox(2);bbox(1)];
interpolate_bbox(:,2) = [bbox(3);bbox(3);bbox(4);bbox(4)];
interpolate_bbox(:,3) = [0;0;0;0];

tcorners=htransform_vectors(inv(hmat),interpolate_bbox);
x = linspace(interpolate_bbox(1,1),interpolate_bbox(2,1),cols);
y = linspace(interpolate_bbox(1,2),interpolate_bbox(4,2),rows);

imclass = class(im);

% now generate the coordinates of a (rows x cols) raster within the frame
% defined by the transformed corners.  x,y and z may all vary along each
% row and column in the general case.  This is just a lot of calls to
% linspace, but is encapsulated in generate_slice_coordinates.

% cnt = bbox2contour(tcorners);
% cnt1 = bbox2contour(im_bbox); cnt1.Color = 'r';
% figure(101); clf; ax_h = gca; hold(ax_h, 'on');
% arbuz_DrawContour3D(ax_h, cnt, eye(1));
% arbuz_DrawContour3D(ax_h, cnt1, eye(1));
% axis(ax_h, 'image'); grid(ax_h, 'on');

[xp,yp,zp]=generate_slice_coordinates(tcorners, cols, rows);

if strcmp(imclass,'uint8')
  myplane = cast(im, 'double');
  %  VI = INTERP3(V,XI,YI,ZI) assumes X=1:N, Y=1:M, Z=1:P
  iplane=interp3(myplane, xp, yp, zp, 'linear',nanval);
  size(iplane)
  imout = cast(iplane, imclass);
else
  imout = interp3(cast(im, 'double'), xp, yp, zp, 'linear',nanval);
end

AAA = hmatrix_translate(-[x(1), y(1), 0])* ...
  hmatrix_scale([(length(x)-1)/(x(end)-x(1)), (length(y)-1)/(y(end)-y(1)), 1]) *...
  hmatrix_translate([1, 1, 0]);
