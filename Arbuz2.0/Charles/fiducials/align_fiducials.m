function [alignmat, rotmat, separations] = align_fiducials(lines, rlines)
%
% finds a rotation transformation that aligns the directions of
% corresponding lines in two lists.  each line is defined by a pair of
% end points, i.e. lines(1,:) to lines(2,:) is a line, etc.
% the number of lines is size(lines, 1) / 2.  So this procedure maximizes
% the projections of unit vectors in the first list ("lines") onto the 
% corresponding unit vectors in the second list ("rlines"), in a least
% square sense.  This is done via the procrustes procedure which matches
% the endpoints of all the unit vectors in a least square sense, as if they
% were sets of matching 3D point fiducials.
%
%
% C. Pelizzari 2007

lunit=unit_vectors(lines);
runit=unit_vectors(rlines);

[alignmat, rotmat, acen, bcen, err] = procrustes(lunit, runit);

tlines=htransform_vectors(alignmat,lines);
npts=size(lines,1);
separations=[];
for i = 1:2:npts
    l1=tlines(i:i+1,:);
    l2=rlines(i:i+1,:);
    separations = [separations;line_to_line_separation(l1,l2)];
end