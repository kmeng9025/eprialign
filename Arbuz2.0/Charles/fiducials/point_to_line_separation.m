function dists = point_to_line_separation(points, line1)
% finds the distance of closest approach between points and a line,
% defined by its endpoints.  
% for point A and line CD, then the computation is:
%
%
% 1) project AD onto the unit vector along CD
% 2) subtract this "along" component from AD leaving
%    the "between" component
%
% 
% 
% C. Pelizzari 2007

uvec1=unit_vectors(line1);

npoints=size(points,1);
dists = zeros(1,npoints);
for n=1:npoints
    fromto=line1(2,:)-points(n,:);
    separation = fromto - uvec1*dot(fromto, uvec1);
    dists(n)=sqrt(dot(separation, separation));
end

