function [dists,closest,alphas] = point_to_poly_1(point, poly,varargin)
% for each face of a polygon, 
% finds the distance of closest approach between "point" and the face,
% defined as the line through its endpoints.  
% for point A and line CD, then the computation is:
%
%
% 1) project AD onto the unit vector along CD
% 2) subtract this "along" component from AD leaving
%    the "between" component
% 
% note that the point of closest approach may be beyond the end of the
% polygon edge, in which case the parameter alpha will be <0 of >1.  alpha
% is the fractional distance along the segment from the starting point to
% the point of closest approach.  we're actually finding the distance to
% the infinite line through the endpoints, not just to the segment.
% 
% C. Pelizzari 2007
sz=size(poly);
if(sz(2)==2),
    zcoord=zeros(sz(1),1);
    poly=[poly zcoord];
    point = [point 0];
end
%size(varargin)
if(size(varargin,2) == 2),
    uvec1=varargin{1};
    len1=varargin{2};
else
    %get unit vectors and lengths of polygon sides.
    [uvec1,len1]=poly_unit_vectors(poly);
end
% note that nsegs is the number of segment starting points, i.e., 1 less
% than the number of points on the calibration curve.
nsegs=numel(len1);
dists = zeros(nsegs,1);
closest = zeros(nsegs,3);
alphas = zeros(nsegs,1);
for n=1:nsegs    
    fromto=point(1,:)-poly(n,:);
    along=dot(fromto,squeeze(uvec1(n,:)));
    closest(n,:) = poly(n,:)+along*uvec1(n,:);
    separation = closest(n,:)-point(1,:);
    dists(n)=dot(separation, separation);
    alphas(n)=along/len1(n);
end

