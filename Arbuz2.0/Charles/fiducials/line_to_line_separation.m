function [separation, dist] = line_to_line_separation(line1, line2)
% finds the distance of closest approach between two lines,
% defined by their endpoints.  
% if the two lines are AB and CD, then the computation is:
%
% if lines are nearly parallel, then
%
% 1) project AD onto the unit vector along CD
% 2) subtract this "along" component from AD leaving
%    the "between component
%
% if lines are not parallel, then
%
% 1) project AD onto direction of mutual normal to AB and CD.
% 
% C. Pelizzari 2007

uvec1=unit_vectors(line1);
uvec2=unit_vectors(line2);

fromto=line2(2,:)-line1(1,:);

if (dot(uvec1, uvec2) > 0.999)
    separation = fromto - uvec1*dot(fromto, uvec1);
    dist = sqrt(dot(separation, separation));
    return;
end

mutual_normal=cross(uvec1,uvec2);

separation = dot(fromto, mutual_normal)*mutual_normal;
dist = sqrt(dot(separation, separation));
%if (dot(separation, fromto) < 0), separation = -separation; end;