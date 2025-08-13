function [newpoints, chisq, hmats] = realign_fiducial_points(pointlist, angles, translations, whichone)

npoints = size(pointlist, 3);
nplanes = size(pointlist, 1);

hmats = zeros(4,4,nplanes);

mypoints = line_up_fiducial(pointlist, whichone);
center = squeeze(mypoints(1,:,whichone));
using=1;
for n = 1:nplanes
    if (n == whichone)
        angle = 0;
    else
        angle = angles(using); using = using + 1;
    end
    hmats(:,:,n) = rotate_about_center(hmatrix_rotate_z(angle), center) ...
        * hmatrix_translate(translations(n,:));
end
newpoints = htransform_planes(mypoints, hmats);

chisq = point_line_residual(newpoints);