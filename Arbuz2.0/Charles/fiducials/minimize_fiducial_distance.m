function [trans, newvecs] = minimize_fiducial_distance(vecs, tvecs);
%
% finds a translation that minimizes the distance between two sets of
% vectors in a least square sense.  
%
% this involves finding the translation whose projection along each of the
% inter-line separation directions is equal to the respective distance.
% Solves by svd to get a least squares solution.
%
% C. Pelizzari 2007

[seps, dists] = line_to_line_separations(tvecs, vecs);

umat = [seps(1,:)/dists(1); seps(2,:)/dists(2); seps(3,:)/dists(3)];

[u,s,v] = svd(umat);
uinv = v * inv(s) * u';

t = uinv * dists;
trans = t';
newvecs=tvecs+repmat(trans, size(tvecs,1), 1);
