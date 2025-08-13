function [Bev_masks] = maskbev(angles , CT_Tumor_mask, CT_hypoxia_mask, mask_out_boost  )

%angles = 0:360/5:360 -(360/5);
antiboost_mask=CT_Tumor_mask - CT_hypoxia_mask;
antiboost_mask(find(antiboost_mask < 0)) = 0;
boost_mask=CT_Tumor_mask&CT_hypoxia_mask;

% bev_mask is the mask volume to which the beams will conform.
% mask_out_boost says whether to block the hypoxic volume or not
% (i.e., antiboost).
% mask_out_boost=1;
bev_mask=boost_mask;

goodpix=find(bev_mask); 
[igood,jgood,kgood]=ind2sub(size(bev_mask),goodpix);
%These 3 lines center the BEV field on the center of mass of the target.
%While this is a good idea, it's easier for us to not make a bed shift if
%we don't have to. I am replacing this orginal code with new code for the
%CENTER of the targeting volume.
icen=mean(igood); %old code for CENTER OF MASS
jcen=mean(jgood);
kcen=mean(kgood);

%NEW code for CENTER OF VOLUME
% icen = round(size(bev_mask,1)/2)
% jcen = round(size(bev_mask,2)/2)
% kcen = round(size(bev_mask,3)/2)

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
scd=234+10;
nangles=numel(angles);
%figure(1); clf
nrows=nangles/5;

if(mask_out_boost),nrows=nrows*3,end
ncols=5;
iplot=1;
[isize, jsize, ksize]=size(CT_Tumor_mask);

xvec=([1:jsize]-jcen)*pixsiz*directions(1);
yvec=([1:isize]-icen)*pixsiz*directions(2);
zvec=([1:ksize]-kcen)*pixsiz*directions(3);

% this will be used later to compute intersection of beamlet centers with
% image volume:
xycorners=[xvec(1) yvec(1); xvec(1) yvec(end); xvec(end) yvec(end); ...
    xvec(end) yvec(1); xvec(1) yvec(1)];  %2D cross section of image volume
xyzcorners=[xycorners zeros(size(xycorners,1),1)];
sourcepos=[sad 0 0];
%nhood=ones(3,3);  % for dilate and erode operations later on
nhood=ones(4,4);  % for dilate and erode operations later on
margin=0.5;  % this is pixels so quite small
nmargin=int32(margin/smallpix);
nhoodmargin=ones(nmargin,nmargin);
totcover=char(0*CT_Tumor_mask);
beam_coverage={};
bevmaps = {};
Bev_masks = cell(1,length(angles));
%matlabpool open 4

%for ii = 1:length(angles)
parfor ii = 1:length(angles)
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
    angle = angles(ii);
    fprintf(1,'Angle = %.0f\n', angle);
    tic;
    disp('Computing BEV map');
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
    xproj=xrot.*magfactor;      % to collimator plane, i.e. < 1.
    yproj=yrot.*magfactor;
    mymap=zeros(1500,1500);
    myicen=size(mymap,1)/2;
    myjcen=size(mymap,2)/2;
    jproj=round(xproj/smallpix+myjcen);
    iproj=round(myicen-yproj/smallpix);
    myxvals=smallpix*([1:size(mymap,2)]-myjcen);
    myyvals=-smallpix*([1:size(mymap,1)]-myicen);
    inds=sub2ind(size(mymap),iproj,jproj);

    mymap(inds)=1;
    numap=imerode(imdilate(mymap,nhood),nhood);
        %numap=imdilate(numap,nhoodmargin);
        disp('masking out boost');
        xyzrot=htransform_vectors(mymat,xyzboost);
        xrot=xyzrot(:,1);               % mask voxels in BEV coordinates
        yrot=xyzrot(:,2); 
        zrot=xyzrot(:,3);

        magfactor=scd./(sad-zrot); % note this is mag from actual point
        xproj=xrot.*magfactor;      % to collimator plane, i.e. < 1.
        yproj=yrot.*magfactor;
        mymap=zeros(1500,1500);
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
        
%         Angle_name = sprintf('%s','Angle_',num2str(angle));
%         Bev_masks.(Angle_name).Boost_map = bmap;
%         Bev_masks.(Angle_name).Antiboost_map = bmap;
         Bev_masks{ii}.Angle = angle;        
         Bev_masks{ii}.Boost_map = bmap;
         Bev_masks{ii}.Antiboost_map = numap;
        
%  clf
% imagesc(Bev_masks{end}.Boost_map)
% isosurface(CT_hypoxia_mask,0.5);
        

%  
% 
%     pause(0.01);
%     toc;
%     
%     test_mask = bmap;
%     
%     % now back project the voxels in the BEV mask back through the image
%     % volume to evaluate coverage.
%     maskpix=find(test_mask);
%     [ipix,jpix]=ind2sub(size(test_mask),maskpix);
%     xpix=myxvals(jpix);
%     ypix=myyvals(ipix);
%     zpix=0*xpix+sad-scd;
%     xyzpix=[xpix' ypix' zpix']; % pixels on collimator plane in BEV coords
%     machpix=htransform_vectors(mymat',xyzpix); %  and in machine coordinates
%     
%     sourcerot=htransform_vectors(gantmat,sourcepos); %source pos in machine
%                                                      %coordinates                                              
%     npts=numel(maskpix);
%     xyzin=[];
%     xyzout=[];
% tic;
%     coverage=0*CT_Tumor_mask;
%     disp('Back Projecting through CT volume');
%     disp('    Computing rays...');
%     for n=1:npts
%         % for each set pixel ("beamlet") in the BEV map, do the following:
%         % - construct a line from the source position at this gantry angle,
%         %   through the BEV pixel's machine coordinates on the collimator
%         %   plane.
%         % - intersect that line with the image volume.  the function
%         %   utl_intersect solves a 2D line-polygon intersection problem,
%         %   returning both the 2D coordinates of the intersections and a
%         %   parametric distance along the line.  We can find the 2D
%         %   parametric distance in the XY gantry rotation plane, and then use
%         %   that to compute the 3D intersections.
%         % - add the intersections (entry and exit of ray into/out of image
%         %   volume) to a list for further processing.
%         [ninter,alpha,xyint]=utl_intersect(xycorners,...
%             sourcerot(1:2),machpix(n,1:2));
%         raydir=machpix(n,:)-sourcerot;
%         if (ninter > 1)
%             xyzin=[xyzin; sourcerot+alpha(1)*raydir];
%             xyzout=[xyzout; sourcerot+alpha(2)*raydir];
%         end
%     end
%     toc;
%     xyzvox=[];
%     nin=size(xyzin,1);
%     dens=3;
%      disp('    Encoding volume...');
%     for n=1:nin
%         % for each pair of entry and exit points of a beamlet through the
%         % image volume, compute a dense track of 3D coordinates from entry
%         % to exit.  These will be converted to a list of irradiated pixels.
%         if ~mod(n,500), fprintf(1,'%d/%d\n',n,nin), end
%         entering=xyzin(n,:);
%         leaving=xyzout(n,:);
%         traveled=leaving-entering;
%         thick=abs(traveled);
%         maxthick=max(thick);
%         nsteps=dens*maxthick/pixsiz;
%         xtrack=linspace(entering(1),leaving(1),nsteps);
%         ytrack=linspace(entering(2),leaving(2),nsteps);
%         ztrack=linspace(entering(3),leaving(3),nsteps);
%         xyzvox=[xtrack' ytrack' ztrack'];
%   
%     if ~isempty(xyzvox),
%         jikpix=int32(htransform_vectors(inv(pix_to_machine),xyzvox)); 
%         goodpix=(jikpix(:,1)<jsize & ...
%             jikpix(:,2)<isize & jikpix(:,3)<ksize);
%         indpix=sub2ind(size(bev_mask),jikpix(goodpix,2),...
%             jikpix(goodpix,1),jikpix(goodpix,3));
%         coverage(unique(indpix))=1;
%     end
%     end
%     %beam_coverage{end+1}=char(coverage);
%     totcover=char(totcover+coverage);
%     toc;
%     pause(0.1);
end

matlabpool close 

end
    