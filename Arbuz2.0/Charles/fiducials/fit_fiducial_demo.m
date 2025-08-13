
% create some fiducials:  two endpoints define each of three fiducials
fiducials = [2 2 0;2 1.75 10;1 -2 0;2 -0.5 10;-1.5 -1 0; -2 0.5 10];
% location of the slice planes where the fiducials will intersect.  this
% will define the points we use in out fitting.
locations = [-0.5 0 0.5 1.0 1.5 2.0 3.0 4.0 5.0];
% a set of angles by which to rotate each of the planes, to misalign the
% points
angles = [0 10 -10 5 20 -20 -20 -30 30];
% a set of translations of each plane, for further misalignment.
translations = [0 0 0;0.8 0.8 0; -0.4 0.6 0; 0.8 -0.6 0; -0.6 -0.8 0; 1 0.9 0;...
    -1 -1 0; 1.2 -0.9 0; 1.6 -1.1 0];
%create the "samples" where the fiducials intersect the planes, and then
%the misaligned version of same
fidpts = resample_fiducials(fiducials, [0 0 1], locations);
misaligned = translate_rotate_planes(fidpts, angles, translations, [0 0 0]);

plot_3d_axes(10);
disp('"ideal" points plotted as polylines');
plot_points_3d(fidpts(:,:,1),'r-');
plot_points_3d(fidpts(:,:,2),'g-');
plot_points_3d(fidpts(:,:,3),'c-');
disp('misaligned points plotted as crosses');
plot_points_3d(misaligned(:,:,1),'r+');
plot_points_3d(misaligned(:,:,2),'g+');
plot_points_3d(misaligned(:,:,3),'c+');

disp('after fitting, realigned points plotted as dots');
pause
options = optimset('TolFun', 0.0001,'MaxFunEvals',500,'TolX',0.0001);
[x,fval] = fminsearch(@(angles)fit_fiducial_plane_points(misaligned,...
    angles, translations * 0,1),zeros(1,prod(size(angles))-1) , options)
[x,fval] = fminsearch(@(angles)fit_fiducial_plane_points(misaligned,...
    angles, translations * 0,1),x , options)
[realigned, chisq] = realign_fiducial_points(misaligned, x, zeros(prod(size(angles)),3), 1);
plot_points_3d(realigned(:,:,1),'r.');
plot_points_3d(realigned(:,:,2),'g.');
plot_points_3d(realigned(:,:,3),'c.');
