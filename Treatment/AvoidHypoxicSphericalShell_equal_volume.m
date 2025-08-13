function [isocenter,results] = AvoidHypoxicSphericalShell_equal_volume(po2image,radius,Mask,center,results)

%%%%% Find sphere targetting hypoxic region, calculate hypoxic tumor +
%%%%% normoxic tumor covered. Then find spherical shell at the same center avoiding
%%%%% target sphere that covers as close to the
%%%%% same amount of tumor volume and as little hypoxic tumor as possible


%[center,results] = TargetHypoxicSphere(po2image,radius,Mask);

totaltumor = results.hypoxic_tumor_insphere_volume + results.normoxic_tumor_insphere_volume;
clear results

tumormask = Mask;

hypoxic_vox =  po2image <= 10 & po2image > -25;
%fitted_vox =  po2image > -25;
normox_vox = po2image > 10;

%tumorvox = numel(find(tumormask == 1));
%intumor = po2image(tumormask == 1);
hypoxicvox = numel(find(hypoxic_vox & tumormask ));
normoxvox = numel(find(normox_vox & tumormask));

centerx = center(1);
centery = center(2);
centerz = center(3);

xxx = zeros(size(po2image));
yyy = zeros(size(po2image));
zzz = zeros(size(po2image));
for ii = 1:size(po2image,1)
    for jj = 1:size(po2image,2)
        for kk = 1:size(po2image,3)
            xxx(ii,jj,kk) = ii;
            yyy(ii,jj,kk) = jj;
            zzz(ii,jj,kk) = kk;
        end
    end
end            
distmat = sqrt((xxx-centerx).^2 + (yyy-centery).^2 + (zzz-centerz).^2);

 [ Inner_radius, Outer_radius ] = Equal_sphere_determination( radius );

outr = Outer_radius;
inr = Inner_radius;
shell1 = zeros(size(po2image));
shell1(distmat <= inr ) = 1;
hypoxictumorinc1 = numel(po2image(shell1 == 1 & tumormask == 1 & po2image <= 10));
normoxictumorinc1 = numel(po2image(shell1 == 1 & tumormask == 1 & po2image > 10));
shell2 = zeros(size(po2image));
shell2(distmat <= outr & distmat > inr) = 1;
hypoxictumorinc2 = numel(po2image(shell2 == 1 & tumormask == 1 & po2image <= 10));
normoxictumorinc2 = numel(po2image(shell2 == 1 & tumormask == 1 & po2image > 10));


results.inradius_1 = inr;
results.outradius_1 = outr;
%results.tumor_inshell_1 = tumorinc(inr);
results.hypoxic_tumor_inshell_fraction_1 = hypoxictumorinc1/hypoxicvox;
results.hypoxic_tumor_inshell_volume_1 = hypoxictumorinc1*(0.668^3);
results.normoxic_tumor_inshell_fraction_1 = normoxictumorinc1/normoxvox ;
results.normoxic_tumor_inshell_volume_1 = normoxictumorinc1*(0.668^3);

%results.inradius_2 = inr(2);
%results.outradius_2 = outr;
% results.tumor_inshell_2 = tumorinc(inr(2));
results.hypoxic_tumor_inshell_fraction = hypoxictumorinc2/hypoxicvox;
results.hypoxic_tumor_inshell_volume = hypoxictumorinc2* (0.668^3);
results.normoxic_tumor_inshell_fraction = normoxictumorinc2/normoxvox;
results.normoxic_tumor_inshell_volume = normoxictumorinc2* (0.668^3);
results.tumor_volume_inshell = hypoxictumorinc2 + normoxictumorinc2*(0.668^3);
results.non_tumor_volume_inshell= numel(find((shell2 == 1 & tumormask ~= 1 & po2image ~= -100)))*(0.668^3);

isocenter = [centerx, centery, centerz];

