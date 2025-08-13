function [xout, yout, zout] = generate_slice_coordinates(corners, columns, rows)
%GENERATE_SLICE_COORDINATES  returns 2D arrays of x, y and z coordinates of pixels 
% on a 3D slice.
%
% inputs:  Corners(4,3)  3D coordinates of the 4 corners of the slice in order
%                        (upper left, upper right, lower right, lower left)
% 
% outputs: xout  (rows x columns) array of x coordinates
%          yout
%          zout
%
% xout, yout, zout can be used as input to "interp3"
% xout, yout can be used as input to "interp2" if we are sure we're on an
% original image plane.


xleft = linspace(corners(1,1), corners(4,1), rows);
yleft = linspace(corners(1,2), corners(4,2), rows);
zleft = linspace(corners(1,3), corners(4,3), rows);

xright = linspace(corners(2,1), corners(3,1), rows);
yright = linspace(corners(2,2), corners(3,2), rows);
zright = linspace(corners(2,3), corners(3,3), rows);

xout=zeros(rows, columns);
yout=zeros(rows, columns);
zout=zeros(rows, columns);

for i = 1:rows 
    xout(i, :) = linspace(xleft(i),xright(i),columns);
    yout(i, :) = linspace(yleft(i),yright(i),columns);
    zout(i, :) = linspace(zleft(i),zright(i),columns);
end
