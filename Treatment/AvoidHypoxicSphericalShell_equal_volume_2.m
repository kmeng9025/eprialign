function [isocenter,results] = AvoidHypoxicSphericalShell_equal_volume_2(po2image,radius,Mask,center,results,resolution)

%%%%% Find sphere targetting hypoxic region, calculate hypoxic tumor +
%%%%% normoxic tumor covered. Then find spherical shell at the same center avoiding
%%%%% target sphere that covers as close to the
%%%%% same amount of tumor volume and as little hypoxic tumor as possible


%[center,results] = TargetHypoxicSphere(po2image,radius,Mask);

totaltumor = results.hypoxic_tumor_insphere_volume + results.normoxic_tumor_insphere_volume;
clear results

tumormask = Mask;
hypoxic_vox =  po2image <= 10 & po2image > -25;
fitted_vox =  po2image > -25;
normox_vox = po2image > 10;

tumormask = Mask&fitted_vox;
nottumormask = (~tumormask) & fitted_vox;

voxel_volume = prod(resolution);
nSize = size(po2image);
[X, Y, Z] = meshgrid(resolution(1)*(1:nSize(1)), resolution(2)*(1:nSize(2)), resolution(3)*(1:nSize(3)));


hypoxicvox = numel(find(hypoxic_vox & tumormask ));
normoxvox = numel(find(normox_vox & tumormask));

centerx = (size(po2image,2)/2)-center(1);
centery = (size(po2image,1)/2)-center(2);
centerz = (size(po2image,3)/2)-center(3);

Icenter = int8(center)

XT = X(Icenter(1),Icenter(2),Icenter(3));
YT = Y(Icenter(1),Icenter(2),Icenter(3));
ZT = Z(Icenter(1),Icenter(2),Icenter(3));
% centerx = center(1);
% centery = center(2);
% centerz = center(3);

 [ Inner_radius, Outer_radius ] = Equal_sphere_determination( radius );

  radiation_mask =  ((X-XT).^2 + (Y-YT).^2 + (Z-ZT).^2) <= Outer_radius^2  & ((X-XT).^2 + (Y-YT).^2 + (Z-ZT).^2) > Inner_radius^2 ; 
  hypoxic_tumor_insphere = numel(find(radiation_mask & tumormask & hypoxic_vox));
  normtiss_insphere= numel(find(radiation_mask & nottumormask));
  normoxic_tumor_insphere = numel(find(radiation_mask & tumormask & normox_vox));
  tumor_outsphere = numel(find(~radiation_mask & tumormask));

% 
% centerx = center(1);
% centery = center(2);
% centerz = center(3);


% shell1 = zeros(size(po2image));
% shell1(distmat <= inr ) = 1;
% hypoxictumorinc1 = numel(po2image(shell1 == 1 & tumormask == 1 & po2image <= 10));
% normoxictumorinc1 = numel(po2image(shell1 == 1 & tumormask == 1 & po2image > 10));


outr = Outer_radius;
inr = Inner_radius;
results.inradius_1 = inr;
results.outradius_1 = outr;
%results.tumor_inshell_1 = tumorinc(inr);
% results.hypoxic_tumor_inshell_fraction_1 = hypoxictumorinc1/hypoxicvox;
% results.hypoxic_tumor_inshell_volume_1 = hypoxictumorinc1*(0.668^3);
% results.normoxic_tumor_inshell_fraction_1 = normoxictumorinc1/normoxvox ;
% results.normoxic_tumor_inshell_volume_1 = normoxictumorinc1*(0.668^3);

%results.inradius_2 = inr(2);
%results.outradius_2 = outr;
% results.tumor_inshell_2 = tumorinc(inr(2));
results.hypoxic_tumor_inshell_fraction = hypoxic_tumor_insphere/hypoxicvox;
results.hypoxic_tumor_inshell_volume = hypoxic_tumor_insphere * (resolution(3)^3);
results.normoxic_tumor_inshell_fraction = normoxic_tumor_insphere/normoxvox;
results.normoxic_tumor_inshell_volume = normoxic_tumor_insphere* (resolution(3)^3);
results.tumor_volume_inshell = (hypoxic_tumor_insphere + normoxic_tumor_insphere)*(resolution(3)^3);
results.non_tumor_volume_inshell= normtiss_insphere*(resolution(3)^3);

isocenter = center;

