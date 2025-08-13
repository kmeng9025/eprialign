function [bevmaps, material_mask, beamtimes, figs] = ...
  maskbev_depths(CT, boost_mask, material_mask, Bev_masks, angles, varargin)

prescription=13;

CTmu=CT;
if max(CTmu(:))>100
  CTmu=(CT+1000)/5000;
end

totmask = material_mask;

material_atten=[0.0206 0.0205 0.029];  % attenuation 1/mm for materials
calibration_atten=0.019;    % measured attenuation for solid water at 225 kVp

target = epr_maskcm(boost_mask);
target = target([2,1,3]);
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
pix_to_machine=hmatrix_translate(-target)...
  * hmatrix_scale(pixsiz*directions);

% consistent with images having column numbers increase toward
% door; row numbers (i direction) increasing down, and planes (k direction)
% increasing away from stage, while machine coordinates are X toward door,
% Y up, Z toward stage.

% also based on assumption that centroid of both_mask has been moved to the
% isocenter, so coordinates are measured from there.

sad=307;
scd=234+10;
nangles=numel(angles);

nrows=ceil(nangles/5);

ncols = min(nangles, 5);
iplot=1;
[isize, jsize, ksize]=size(CT);

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
title(sprintf('target (%i,%i,%i)',target(1),target(2),target(3)))

maskfig=figure;
% this will be used later to compute intersection of beamlet centers with
% image volume:
sourcepos=[sad 0 0];
bevmaps = {};
beam_areas=[];
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
  numap = Bev_masks{nangle}.Dilated_boost_map; %Add so that mymap gets the correct size. slightly reduntant.
  gantmat=hmatrix_rotate_z(angle);
  
  mymap=zeros(size(numap));
  myicen=size(mymap,1)/2;
  myjcen=size(mymap,2)/2;
  myxvals=smallpix*((1:size(mymap,2))-myjcen);
  myyvals=-smallpix*((1:size(mymap,1))-myicen);
  
  numap = Bev_masks{nangle}.Dilated_boost_map;
  
  figure(maskfig);
  subplot(nrows,ncols,iplot);
  imagesc(numap,'XData',myxvals,'YData',myyvals);
  set(gca,'ydir','normal');
  %plot(xproj,yproj,'r.');
  axis image
  title(['thet = ' num2str(angle)]);
  bevmaps{end+1} = Bev_masks{nangle}.Dilated_boost_map;
  %     bevmaps{end+1} = numap;
  % area of open aperture at isocenter - for computing monitor units
  beam_areas(nangle)=numel(find(numap(:)))*smallpix^2*sad/scd;
  equivalent_circle(nangle)=2*sqrt(beam_areas(nangle)/pi());
  
  pause(0.01);
  iplot=iplot+1;
  toc;
  
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
  
end
fprintf('angle\tgeo depth\tSW path\trad path\teff depth\t  area \teq dia\n');
for n=1:numel(angles)
  fprintf('%5.0f\t%9.2f\t%7.4f\t%8.4f\t%9.2f\t%7.3f\t%10.2f\n', ...
    angles(n),geodepth(n),geopath(n),radpath(n),effdepth(n),...
    beam_areas(n),equivalent_circle(n));
end
depth_info = struct('angles',angles,'geo_depth',geodepth,'geopath',geopath,...
  'radpath',radpath,'effdepth',effdepth,'beam_areas',beam_areas,...
  'equivalent_circle',equivalent_circle,'dose_rate',0);
beamtimes=0*angles;
material_mask=totmask;
if prescription >0
  disp('computing beam on times');
  %[fname,pathname]=uigetfile('*.dosedata','Select dose data file');
  [Col_Type,kV,mA,Filter,Depths,DoseRates,FieldSizes,Col_Description,...
    EndEff,Comment]=CollimatorDoseRates('z:\CenterMATLAB\3dparty\FromChuck','UCCollimators_Circular_225kVp_6_2016.dosedata');
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

figs = [imfig, maskfig];
