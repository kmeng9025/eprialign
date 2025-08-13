function [separations, dists] = line_to_line_separations(lines, rlines)
% finds separations between lines defined by pairs of points.
% lines and rlines are Nx3, each row is a point in 3-space.
%
% first line AB goes from lines(1,:) to lines (2,:)
% its corresponding line CD goes from rlines(1,:) to rlines(2,:)
% etc.
% separations are computed by projecting a vector connecting any point on 
% AB to any point on CD, onto their mutual normal which will be
% the direction of the vector connecting their points of closest approach.
% that calculation is in function line_to_line_separation; this one just 
% calls that one multiple times.
%
% C. Pelizzari 2007

npts=size(lines,1);
separations=[];
dists = [];
for i = 1:2:npts
    l1=lines(i:i+1,:);
    l2=rlines(i:i+1,:);
    [separation, dist] = line_to_line_separation(l1,l2);
    separations = [separations;separation];
    dists = [dists; dist];
end