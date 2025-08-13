function [BEV, beam_target, BEV_volume] = epr_maskbev(angle, the_mask, beam_target)

% beam_center
jcen = beam_target(1);
icen = beam_target(2);
kcen = beam_target(3);

pars = [];
ImagePix = safeget(pars, 'ImagePix', 0.1);     % image pixel size [mm]
PlugPix  = safeget(pars, 'PlugPix', 0.025);    % aperture pix [mm]
PlugSize = safeget(pars, 'PlugSize', [1500 1500]);
sad  = safeget(pars, 'sad', 307);    % [mm]
scd  = safeget(pars, 'scd', 234+10); % [mm]

directions=[1 -1 -1];
pix_to_machine=hmatrix_translate(-[jcen icen kcen])...
  * hmatrix_scale(ImagePix*directions);

% consistent with images having column numbers increase toward
% door; row numbers (i direction) increasing down, and planes (k direction)
% increasing away from stage, while machine coordinates are X toward door,
% Y up, Z toward stage.

% also based on assumption that centroid of both_mask has been moved to the
% isocenter, so coordinates are measured from there.

goodpix=find(the_mask);
[igood,jgood,kgood]=ind2sub(size(the_mask),goodpix);
xyz=[jgood igood kgood];
xyz=htransform_vectors(pix_to_machine, xyz);

tic;
fprintf('Computing BEV map for angle %i\n', angle);
%gantmat=hmatrix_rotate_z(angle); %Angle is for CW rotation which is slightly counterintuitive to the user.
gantmat=hmatrix_rotate_z(angle); %-Angle is for CCW rotation which is slightly better for the user.
%gantmat=hmatrix_rotate_z(-angle+180); %-Angle is for CCW rotation which is slightly better for the user. Add 180 to flip the whole procedure.
bevmat=hmatrix_rotate_y(90);
mymat=gantmat'*bevmat';
xyzrot=htransform_vectors(mymat,xyz);
xrot=xyzrot(:,1);               % mask voxels in BEV coordinates
yrot=xyzrot(:,2);
zrot=xyzrot(:,3);

magfactor=scd./(sad-zrot); % note this is mag from actual point
xproj=xrot.*magfactor;     % to collimator plane, i.e. < 1.
yproj=yrot.*magfactor;

BEV=zeros(PlugSize(1),PlugSize(2));
myicen=size(BEV,1)/2;
myjcen=size(BEV,2)/2;
jproj=round(xproj/PlugPix+myjcen);
iproj=round(myicen-yproj/PlugPix);
inds=sub2ind(size(BEV),iproj,jproj);
BEV(inds)=1;

% this will be used to compute intersection of beamlet centers with
% image volume for dilate and erode operations

nhood=ones(4,4);  
BEV=imerode(imdilate(BEV,nhood),nhood);

BEV_volume = numel(find(BEV)) * (PlugPix^2);



