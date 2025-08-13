function resliced_volume=reslice_volume(hmat1, hmat2, planevol, imagevol,...
    nullvalue, direction)
%
% reslices one image volume along the planes of another, using matrices
% saved by register_manual.  The saved matrices contain all the data
% necessary to map points back and forth from one image volume to the
% other.  The transformation is invertible, i.e. can be used in either
% direction.  However, the order of the two matrices does matter, and there
% is a "forward" and a "reverse" direction.  
%
% for a "forward" reslicing (direction >= 0):
%
% planevol (the first volume) defines the planes of the new volume
% imagevol (the second volume) is the image data to be resliced
% 
% for a "reverse" reslicing (direction < 0):
%
% planevol (first volume) is the image data
% imagevol (second volume) defines the planes of the new volume
%
% You must enter the volumes in the same order you did when you ran
% register_manual to generate the registration matrix.  For example, if you
% registered EPRI with MRI as follows:
%
% register_manual(mrivol, 'scale',mriscale,eprvol,'scale',eprscale)
%
% then you must run reslice_volume(hmat1, hmat2, mrivol, eprvol,...)
% to get correct results.  If you want to reslice the EPR onto the MRI
% planes, this would be a "forward" reslicing, i.e. direction=0 or 1.  If
% you want to reslice the MRI onto the EPR planes, this is a "reverse"
% reslicing, i.e. direction = -1.
%
% Voxels in the new volume that do not intersect the image data volume
% get the value "nullvalue".  You can set this to zero, or some large
% positive or negative value, depending on what you intend to do with the
% result.
%
% you can get the matrices from the saved registration file by using
%
% [hmat1,hmat2]=read_registration_matrix();
%
% note that this can be used to reslice not only actual image data, but
% also masks that represent regions of interest, as long as they are in the
% same pixel coordinate system as the image.
%
% C. Pelizzari August 2005

corners=zeros(4,3);



%hmat1
%hmat2
if isempty(hmat1) | isempty(hmat2)
    %user can pass empty hmat1 and hmat2 and be prompted for the matrix
    %file CH 8-9-06
    disp('Select the registration matrix file')
    [hmat1,hmat2]=read_registration_matrix();
end
    
if (direction >= 0)
    %forward direction jdim --> idim
    idim=size(planevol);
    jdim=size(imagevol);
    mymat=inv(hmat1)*hmat2;
else
    %reverse direction idim --> jdim
    idim=size(imagevol);
    jdim=size(planevol);
    mymat=inv(inv(hmat1)*hmat2);
end
resliced_volume=zeros(idim);

% switched 1st and 2nd dimension because anisotropic images don't
% work  CH 5-24-06
[xvol yvol zvol] = generate_image_coordinates(1, jdim(2), jdim(2),...
    1, jdim(1), jdim(1), 1, jdim(3), jdim(3));

for slice_t = 1:idim(3)
% switched 1st and 2nd dimension because anisotropic images don't
% work  CH 5-24-06
    corners(:,:) = [[1 1 slice_t];[idim(2) 1 slice_t]; ...
        [idim(2) idim(1) slice_t]; [1 idim(1) slice_t]];
    [xs ys zs] = generate_slice_coordinates(htransform_vectors(mymat,corners),...
        idim(2), idim(1));
    
    if (direction >= 0)
        reslice=interp3(xvol, yvol, zvol, imagevol, xs, ys, zs);
    else
        reslice=interp3(xvol, yvol, zvol, planevol, xs, ys, zs);
    end
    reslice(find(isnan(reslice)))=nullvalue;


    % added 4-21-05 CAP - mask off pixels below threshold in resliced
    % image

    resliced_volume(:,:,slice_t)=reslice;
end

