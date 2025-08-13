function [isocenter,results] = AvoidHypoxicSphericalShell(po2image,radius,Mask)

%%%%% Find sphere targetting hypoxic region, calculate hypoxic tumor +
%%%%% normoxic tumor covered. Then find spherical shell at the same center avoiding
%%%%% target sphere that covers as close to the
%%%%% same amount of tumor volume and as little hypoxic tumor as possible


[center,results] = TargetHypoxicSphere(po2image,radius,Mask);

totaltumor = results.hypoxic_tumor_insphere_volume + results.normoxic_tumor_insphere_volume;
clear results

tumormask = Mask;

tumorvox = numel(find(tumormask == 1));
intumor = po2image(tumormask == 1);
hypoxicvox = numel(find(intumor <= 10));
normoxvox = numel(find(intumor > 10));

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

maxradius = 20;
check = 0;
for outradius = 1:maxradius
    tempsphere = zeros(size(po2image));
    tempsphere(distmat <= outradius) = 1;
    tumorinc(outradius) = numel(po2image(tempsphere == 1 & tumormask == 1));
    if  outradius > 1 && tumorinc(outradius) == tumorinc(outradius-1) && check == 0
        outr = outradius - 1;
        check = check + 1;
    end
end

check = 0;
diff = [];
tumorinc = [];
for inradius = 1:outr
    tempshell = zeros(size(po2image));
    tempshell(distmat <= outr & distmat > inradius) = 1;
    tumorinc(inradius) = numel(po2image(tempshell == 1 & tumormask == 1));
    diff(inradius) = tumorinc(inradius) - totaltumor;
    if diff(inradius) < 0 
        check = check + 1;
    end
    if check == 1 && inradius > 1
        inr(1) = inradius-1;
        inr(2) = inradius;
    elseif check == 1 && inradius == 1
        inr(1) = inradius;
        inr(2) = inradius;        
    end
end

shell1 = zeros(size(po2image));
shell1(distmat <= outr & distmat > inr(1)) = 1;
hypoxictumorinc1 = numel(po2image(shell1 == 1 & tumormask == 1 & po2image <= 10));
normoxictumorinc1 = numel(po2image(shell1 == 1 & tumormask == 1 & po2image > 10));
shell2 = zeros(size(po2image));
shell2(distmat <= outr & distmat > inr(2)) = 1;
hypoxictumorinc2 = numel(po2image(shell2 == 1 & tumormask == 1 & po2image <= 10));
normoxictumorinc2 = numel(po2image(shell2 == 1 & tumormask == 1 & po2image > 10));


results.inradius_1 = inr(1);
results.outradius_1 = outr;
results.tumor_inshell_1 = tumorinc(inr(1));
results.hypoxic_tumor_inshell_fraction_1 = hypoxictumorinc1/hypoxicvox;
results.hypoxic_tumor_inshell_volume_1 = hypoxictumorinc1;
results.normoxic_tumor_inshell_fraction_1 = normoxictumorinc1/normoxvox;
results.normoxic_tumor_inshell_volume_1 = normoxictumorinc1;

results.inradius_2 = inr(2);
results.outradius_2 = outr;
results.tumor_inshell_2 = tumorinc(inr(2));
results.hypoxic_tumor_inshell_fraction_2 = hypoxictumorinc2/hypoxicvox;
results.hypoxic_tumor_inshell_volume_2 = hypoxictumorinc2;
results.normoxic_tumor_inshell_fraction_2 = normoxictumorinc2/normoxvox;
results.normoxic_tumor_inshell_volume_2 = normoxictumorinc2;

isocenter = [centerx, centery, centerz];

