function [xout, yout, zout] = generate_image_coordinates(xlo, xhi, xcount, ylo, yhi, ycount, zlo, zhi, zcount)
%GENERATE_IMAGE_COORDINATES  returns 3D arrays of x, y and z coordinates of image pixels
%
% inputs:  xlo, xhi, xcount   min and max x coordinate, number of pixels 
%          ylo, yhi, ycount   can take account of pixel size if you want.
%          zlo, zhi, zcount
% x is assumed to lie across image ROWS (second array index)
% y down image COLUMNS (first array index)
% z along image PAGE direction (third array index)
%
% outputs:  xout  3D array of x values - dimensions are (ycount, xcount, zcount).
%           yout
%           zout
%
% xout, yout, zout can be used as input to "interp3"


%xout = zeros(ycount, xcount, zcount);
%yout = zeros(ycount, xcount, zcount);
%zout = zeros(ycount, xcount, zcount);

xrow = linspace(xlo, xhi, xcount);
ycol = linspace(ylo, yhi, ycount);
zstack=linspace(zlo,zhi,zcount);

[yout xout zout] = ndgrid(ycol, xrow, zstack);

%ycol = linspace(ylo, yhi, ycount)';
%zstack=linspace(zlo,zhi,zcount)';

%xplane=repmat(xrow,[ycount 1]);
%yplane=repmat(ycol,[1 xcount]);

%xout=repmat(xplane,[1 1 zcount]);
%yout=repmat(yplane,[1 1 zcount]);
%for i = 1: zcount
 %   zout(:,:,i)=repmat(zstack[i],[ycount xcount]]);
 %end;


