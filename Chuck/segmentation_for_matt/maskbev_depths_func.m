function [bevmaps, material_mask, depth_info, varargout] = ...
   maskbev_depths_func(CT, CT_Tumor_mask, CT_hypoxia_mask, angles, ...
     varargin)
%MASKBEV_DEPTHS_FUNC:  computes BEV masks and beam times required to
% deliver prescribed dose.
%
% [bevmaps, material_mask, depth_info, varargout] = ...
%    maskbev_depths_func(CT, CT_Tumor_mask, CT_hypoxia_mask, angles, ...
%      varargin)
%
% Inputs:
% CT:  3D array of CT values.  If these are integers with large values,
%   they will be converted to nominal attenuation coefficients by 
%   mu=(CT+1000)/5000.
% CT_Tumor_mask:  logical array, same size as CT
% CT_hypoxia_mask: logical array
% angles:  vector of beam angles
% (optional) prescription:  Gray prescribed at isocenter.  if given, beam
%    times will be returned as 4th output 
% (optional) docoverage:  if =1, coverage maps will be returned as 5th
%    output
% (optional) mask_out_boost:  if =1, do calculation for antiboost vs boost
% 
% Ouputs:
% bevmaps:  cell array with beam apertures for each angle
% material_mask:  array of size(CT) with values 1 for skin, 2 for acrylic
%   ring, 3 for PVS cushion
% depth_info:  structure with fields "angles", "geo_depth","geopath",
%   "radpath", "effdepth", "areas", "equivalent_circle" for each beam.
%   "effdepth" is the pathlength in solid water equivalent to the actual
%   radiographic pathlength through the three materials, for each beam.  
% (optional) beamtimes: if prescription >= 0, times for each beam to
%   deliver "prescription" Gray at isocenter.
% (optional) beam_coverage: cell array of coverage maps for each beam
%
% C. Pelizzari, Aug 2016
%


antiboost_mask=CT_Tumor_mask - CT_hypoxia_mask;
antiboost_mask(find(antiboost_mask < 0)) = 0;
boost_mask=CT_Tumor_mask&CT_hypoxia_mask;

prescription=-1;
if nargin > 4, prescription = varargin{1}; end
 docoverage=0;
if nargin > 5, 
    docoverage=varargin{2}; 
end
mask_out_boost = 0;
if nargin > 6, mask_out_boost = varargin{3}; end

oldway=1;
if oldway,
    CTmu=CT;
    if max(CTmu(:))>100, 
        CTmu=(CT+1000)/5000;
    end
    [skinmask,ringmask,pvmask]=findskin(CTmu);
    totmask=skinmask+2*ringmask+3*pvmask;
else
    CT_data=CT;
    fit_cylinder_points_gaussian;
    totmask=legmask+2*ringmask+3*pvsmask;
end

material_atten=[0.0206 0.0205 0.029];  % attenuation 1/mm for materials
calibration_atten=0.019;    % measured attenuation for solid water at 225 kVp

% bev_mask is the mask volume to which the beams will conform.
% mask_out_boost says whether to block the hypoxic volume or not
% (i.e., antiboost).
mask_out_boost=0;
bev_mask=boost_mask;

goodpix=find(bev_mask); 
[igood,jgood,kgood]=ind2sub(size(bev_mask),goodpix);
itgt=round(mean(igood));
jtgt=round(mean(jgood));
ktgt=round(mean(kgood));

icen=itgt;
jcen=jtgt;
kcen=ktgt;

pixsiz=0.1;
smallpix=0.025;

directions=[1 -1 -1];
pix_to_machine=hmatrix_translate(-[jcen icen kcen])...
    * hmatrix_scale(pixsiz*directions);

% consistent with images having column numbers increase toward
% door; row numbers (i direction) increasing down, and planes (k direction)
% increasing away from stage, while machine coordinates are X toward door, 
% Y up, Z toward stage.

% also based on assumption that centroid of both_mask has been moved to the
% isocenter, so coordinates are measured from there.

xyz=[jgood igood kgood];                                          
xyz=htransform_vectors(pix_to_machine, xyz);


if mask_out_boost,
    boostpix=find(boost_mask); 
    [iboost,jboost,kboost]=ind2sub(size(boost_mask),boostpix);
    xyzboost=[jboost iboost kboost];
    xyzboost=htransform_vectors(pix_to_machine, xyzboost);
end
sad=307;
scd=234;
nangles=numel(angles);

nrows=ceil(nangles/5);

if(mask_out_boost),nrows=nrows*3,end
ncols=5;
iplot=1;
[isize, jsize, ksize]=size(CT_Tumor_mask);

xvec=([1:jsize]-jcen)*pixsiz*directions(1);
yvec=([1:isize]-icen)*pixsiz*directions(2);
zvec=([1:ksize]-kcen)*pixsiz*directions(3);
[xgrid,ygrid,zgrid]=meshgrid(xvec,yvec,zvec);

imfig=figure;
midmask=squeeze(totmask(:,:,ktgt));
subplot(1,2,1), imagesc(midmask),axis image
colormap(gca,jet);
title('material masks');

subplot(1,2,2);
midslice=CTmu(:,:,ktgt);
imagesc(midslice,'XData',xvec,'YData',yvec), axis image, colormap(gca, gray)
set(gca,'ydir','normal');
target=[jtgt itgt ktgt];
tgtxyz=htransform_vectors(pix_to_machine,target);
hold on
plot(tgtxyz(1),tgtxyz(2),'+');

maskfig=figure;
% this will be used later to compute intersection of beamlet centers with
% image volume:
xycorners=[xvec(1) yvec(1); xvec(1) yvec(end); xvec(end) yvec(end); ...
    xvec(end) yvec(1); xvec(1) yvec(1)];  %2D cross section of image volume
xyzcorners=[xycorners zeros(size(xycorners,1),1)];
sourcepos=[sad 0 0];
nhood=ones(3,3);  % for dilate and erode operations later on
margin=0.5;  % this is pixels so quite small
nmargin=int32(margin/smallpix);
nhoodmargin=ones(nmargin,nmargin);
totcover=char(0*CT_Tumor_mask);
beam_coverage={};
bevmaps = {};
beam_areas=[];
equivalent_radius=[];
geopath=[];
geodepth=[];
radpath=[];
effdepth=[];

for nangle=1:numel(angles)
    angle=angles(nangle);
    % XRad gantry angle is zero when source is on +X axis (source at door).
    % gantry +90 has source at top pointing down, which is a CCW 90 deg
    % rotation from zero.  this is consistent with normal angular
    % conventions, so to rotate points to any specified gantry angle we
    % can just rotate by gantry angle about Z.  
    
    % what we want is to view and project points from a specifed gantry 
    % angle's BEV, with Z pointing from isocenter toward source.  The BEV
    % should have X increasing away from the stage, so as we look at the
    % BEV image, the right of the image (+X) is toward the foot of the bed.
    % We can get this by first rotating about Z by the gantry angle, and
    % then about Y by 90 degrees.  Note that our hmats rotate POINTS, not
    % coordinate systems; so we need to apply the inverse of each of the
    % above transformations
    fprintf(1,'Angle = %.0f\n', angle);
    tic;
    disp('Computing BEV map');
    gantmat=hmatrix_rotate_z(angle);
    bevmat=hmatrix_rotate_y(90);
    mymat=gantmat'*bevmat';
    xyzrot=htransform_vectors(mymat,xyz);
    xrot=xyzrot(:,1);               % mask voxels in BEV coordinates
    yrot=xyzrot(:,2); 
    zrot=xyzrot(:,3);
    
    magfactor=scd./(sad-zrot); % note this is mag from actual point
    xproj=xrot.*magfactor;      % to collimator plane, i.e. < 1.
    yproj=yrot.*magfactor;
    mymap=zeros(500,500);
    myicen=size(mymap,1)/2;
    myjcen=size(mymap,2)/2;
    jproj=round(xproj/smallpix+myjcen);
    iproj=round(myicen-yproj/smallpix);
    myxvals=smallpix*([1:size(mymap,2)]-myjcen);
    myyvals=-smallpix*([1:size(mymap,1)]-myicen);
    inds=sub2ind(size(mymap),iproj,jproj);

    mymap(inds)=1;
    numap=imerode(imdilate(mymap,nhood),nhood);
    if mask_out_boost,
        %numap=imdilate(numap,nhoodmargin);
        disp('masking out boost');
        xyzrot=htransform_vectors(mymat,xyzboost);
        xrot=xyzrot(:,1);               % mask voxels in BEV coordinates
        yrot=xyzrot(:,2); 
        zrot=xyzrot(:,3);

        magfactor=scd./(sad-zrot); % note this is mag from actual point
        xproj=xrot.*magfactor;      % to collimator plane, i.e. < 1.
        yproj=yrot.*magfactor;
        mymap=zeros(500,500);
        myicen=size(mymap,1)/2;
        myjcen=size(mymap,2)/2;
        jproj=round(xproj/smallpix+myjcen);
        iproj=round(myicen-yproj/smallpix);
        myxvals=smallpix*([1:size(mymap,2)]-myjcen);
        myyvals=-smallpix*([1:size(mymap,1)]-myicen);
        inds=sub2ind(size(mymap),iproj,jproj);

        mymap(inds)=1;
        antimap=numap;
        bmap=imerode(imdilate(mymap,nhood),nhood);
        numap=antimap - bmap;
        numap(find(numap<0)) = 0;
    end
    figure(maskfig);
    subplot(nrows,ncols,iplot);
    imagesc(numap,'XData',myxvals,'YData',myyvals);
    set(gca,'ydir','normal');
    %plot(xproj,yproj,'r.');
    axis image
    title(['thet = ' num2str(angle)]);
    bevmaps{end+1} = numap;
    % area of open aperture at isocenter - for computing monitor units
    beam_areas(nangle)=numel(find(numap(:)))*smallpix^2*sad/scd;
    equivalent_circle(nangle)=2*sqrt(beam_areas(nangle)/pi());
    if mask_out_boost,
        njump=nangles/5*ncols;
        subplot(nrows,ncols,iplot+njump);
        imagesc(antimap,'XData',myxvals,'YData',myyvals);
        set(gca,'ydir','normal');
        %plot(xproj,yproj,'r.');
        axis image
        subplot(nrows,ncols,iplot+njump+ncols);
        imagesc(bmap,'XData',myxvals,'YData',myyvals);
        set(gca,'ydir','normal');
        %plot(xproj,yproj,'r.');
        axis image
    end
 

    pause(0.01);
    iplot=iplot+1;
    toc;
    % now back project the voxels in the BEV mask back through the image
    % volume to evaluate coverage.
    maskpix=find(numap);
    [ipix,jpix]=ind2sub(size(numap),maskpix);
    xpix=myxvals(jpix);
    ypix=myyvals(ipix);
    zpix=0*xpix+sad-scd;
    xyzpix=[xpix' ypix' zpix']; % pixels on collimator plane in BEV coords
    machpix=htransform_vectors(mymat',xyzpix); %  and in machine coordinates
    
    sourcerot=htransform_vectors(gantmat,sourcepos); %source pos in machine
                                                     %coordinates      
    dens=3;                                                   
    % compute depth from source to isocenter along this direction -
    % intersect ray from source to iso with the "totmask" volume, getting
    % the number of voxels occupied by each type of material (skin, ring,
    % pvs).  depth is then the length through each of these materials,
    % times its attenuation coefficient, divided by the attenuation
    % coefficient of the medium in which dosimetry was done (solid water)
    
   % disp('Finding central axis depth for this beam');
    
    centralray=tgtxyz-sourcerot;
    traveled=sqrt(sum(centralray.*centralray));
    nsteps=traveled*dens/pixsiz;  %number of steps to divide up this ray
    xtrack=linspace(sourcerot(1),tgtxyz(1), nsteps);
    ytrack=linspace(sourcerot(2),tgtxyz(2), nsteps);
    ztrack=linspace(sourcerot(3),tgtxyz(3), nsteps);
    raymaskvals=interp3(xgrid,ygrid,zgrid,totmask,xtrack,ytrack,ztrack,...
        'linear',0);
    raysum=0;
    seglimits=[0.1 1.5; 1.5 2.5; 2.5 3.5];
    geodepth(nangle)=numel(find(raymaskvals>=min(seglimits(:))))*pixsiz/dens;
    geopath(nangle)=geodepth(nangle)*calibration_atten;
    plotcolors={'c.', 'y.', 'r.'};
    for i=1:3
        thesepixels=find(raymaskvals>=seglimits(i,1) & raymaskvals<seglimits(i,2));
        figure (imfig); subplot(1,2,2);
        hold on;
        plot(xtrack(thesepixels),ytrack(thesepixels),plotcolors{i});
        thismaterial=numel(thesepixels);
        thispath=thismaterial*pixsiz/dens;
        raysum=raysum+thispath*material_atten(i);
    end
    radpath(nangle)=raysum;
    effdepth(nangle)=raysum/calibration_atten;
   
    npts=numel(maskpix);
    xyzin=[];
    xyzout=[];
tic;
docoverage=0;
    if docoverage,
    coverage=0*CT_Tumor_mask;
    disp('Back Projecting through CT volume');
    disp('    Computing rays...');
    for n=1:npts
        % for each set pixel ("beamlet") in the BEV map, do the following:
        % - construct a line from the source position at this gantry angle,
        %   through the BEV pixel's machine coordinates on the collimator
        %   plane.
        % - intersect that line with the image volume.  the function
        %   utl_intersect solves a 2D line-polygon intersection problem,
        %   returning both the 2D coordinates of the intersections and a
        %   parametric distance along the line.  We can find the 2D
        %   parametric distance in the XY gantry rotation plane, and then use
        %   that to compute the 3D intersections.
        % - add the intersections (entry and exit of ray into/out of image
        %   volume) to a list for further processing.
        % 
        % NOTE:
        % for dose calculation, intersect with body contour instead of
        % image volume.  Since body contour is not a rectangular solid,
        % this will require 3D treatment of the intersections, vs. 2D as is
        % being done here.  maybe more economical to do this by masking
        % with the body contour later, rather than bother with it here.
        [ninter,alpha,xyint]=utl_intersect(xycorners,...
            sourcerot(1:2),machpix(n,1:2));
        raydir=machpix(n,:)-sourcerot;
        % here we could transform back to voxel coordinates, and mark all
        % voxels touched by this ray from xyzin to xyzout.  then, mask that
        % list against the body contour.  Then, save that masked list of
        % voxels as the locations along this ray where dose needs to be
        % computed. The voxel marking currently performed in the second
        % loop (for n=1:nin) below would then take place here.
        if (ninter > 1)
            xyzin=[xyzin; sourcerot+alpha(1)*raydir];
            xyzout=[xyzout; sourcerot+alpha(2)*raydir];
        end
    end
    toc;
    xyzvox=[];
    nin=size(xyzin,1);
  
     disp('    Encoding volume...');
    for n=1:nin
        % for each pair of entry and exit points of a beamlet through the
        % image volume, compute a dense track of 3D coordinates from entry
        % to exit.  These will be converted to a list of irradiated pixels.
        if ~mod(n,500), fprintf(1,'%d/%d\n',n,nin), end
        entering=xyzin(n,:);
        leaving=xyzout(n,:);
        traveled=leaving-entering;
        thick=abs(traveled);
        maxthick=max(thick);
        nsteps=dens*maxthick/pixsiz;
        xtrack=linspace(entering(1),leaving(1),nsteps);
        ytrack=linspace(entering(2),leaving(2),nsteps);
        ztrack=linspace(entering(3),leaving(3),nsteps);
        xyzvox=[xtrack' ytrack' ztrack'];

    if ~isempty(xyzvox),
        jikpix=int32(htransform_vectors(inv(pix_to_machine),xyzvox)); 
        % here could also mask against the body contour.  We know these
        % points are ordered from entrance to exit, due to the way we
        % generated them.  So, we could compute the distances to each of
        % them, take the first one as the surface and compute depth and
        % distance to each succeeding one.  apply a Burns correction to the
        % %dd interpolated from the beam data file and record the dose.
        goodpix=(jikpix(:,1)<jsize & ...
            jikpix(:,2)<isize & jikpix(:,3)<ksize);
        indpix=sub2ind(size(bev_mask),jikpix(goodpix,2),...
            jikpix(goodpix,1),jikpix(goodpix,3));
        coverage(unique(indpix))=1;
    end
    end
    beam_coverage{end+1}=char(coverage);
    totcover=char(totcover+coverage);
    toc;
    pause(0.1);
    end

end

depth_info = struct('angles',angles,'geodepth',geodepth,'geopath',geopath,...
    'radpath',radpath,'effdepth',effdepth,'beam_areas',beam_areas,...
    'equivalent_circle',equivalent_circle,'dose_rate',0);
fprintf('angle\tgeo depth\tSW path\trad path\teff depth\t  area \teq dia\n');
for n=1:numel(angles)
    fprintf('%5.0f\t%9.2f\t%7.4f\t%8.4f\t%9.2f\t%7.3f\t%6.2f\n', ...
        depth_info.angles(n),depth_info.geodepth(n),depth_info.geopath(n),...
        depth_info.radpath(n),depth_info.effdepth(n),...
        depth_info.beam_areas(n),depth_info.equivalent_circle(n));
end
beamtimes=0*angles;
material_mask=totmask;
if prescription >0 ,
    disp('computing beam on times');
    [fname,pathname]=uigetfile('*.dosedata','Select dose data file');
    [Col_Type,kV,mA,Filter,Depths,DoseRates,FieldSizes,Col_Description,...
        EndEff,Comment]=CollimatorDoseRates(pathname,fname);
    fprintf('Collimator type: %s; kV, mA = %.1f %.2f, %s\ncomment = %s\n',...
        Col_Type{1}, kV, mA, Filter, Comment);
    [rgrid,dgrid]=meshgrid(FieldSizes,Depths);
    for n=1:nangles
        depth=depth_info(1).effdepth(n)/10;
        colsize=depth_info(1).equivalent_circle(n)/10;
        doserate=interp2(rgrid,dgrid,DoseRates,colsize,depth,'spline');
        depth_info.dose_rate(n)=doserate;
        beamtimes(n)=60*prescription/nangles/doserate+EndEff;
    end
end
if nargout > 3, varargout{1}=beamtimes; end
if docoverage && nargout>4,
    vargout{2}=beam_coverage;
end
    