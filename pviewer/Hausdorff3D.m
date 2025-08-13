% An alternative Hausdorff function for 'similar' masks
% HH is the 'maximum' Hausdorff distance
% vec is average Hasdorff offset between voxels which are offseted

function [HH, vec] = Hausdorff3D(Mask1, Mask2)

idx1 = find(Mask1);
idx2 = find(Mask2);

[X,Y,Z] = meshgrid(1:64,1:64,1:64);

n1 = numel(idx1);
n2 = numel(idx2);

if n1 ==0 || n2 == 0
    HH = 0;
    return
end

% D = zeros(n1, n2);
XX = zeros(n1, n2);
YY = zeros(n1, n2);
ZZ = zeros(n1, n2);
HDx = zeros(n1,1);
HDy = zeros(n1,1);
HDz = zeros(n1,1);

for ii=1:n1
    for jj=1:n2
        XX(ii,jj) = X(idx1(ii)) - X(idx2(jj));
        YY(ii,jj) = Y(idx1(ii)) - Y(idx2(jj));
        ZZ(ii,jj) = Z(idx1(ii)) - Z(idx2(jj));
    end
end

D = sqrt(XX.^2 + YY.^2 + ZZ.^2);

[HD1, IHD1] = min(D,[],1);
[HD2, IHD2] = min(D,[],2);
HH = max(max(HD1), max(HD2));

for ii=1:n1
    [HDD(ii),HDidx(ii)] = min(D(ii,:));
    HDx(ii) = XX(ii,HDidx(ii));
    HDy(ii) = YY(ii,HDidx(ii));
    HDz(ii) = ZZ(ii,HDidx(ii));
end


% calculation way number 2 a bit closer to the description
[HD1_op2_1, IHDX_op2_1] = max(D,[],1);
[~, IHDX_op2_2] = max(HD1_op2_1);
HHop2 = max(min(D(IHDX_op2_1(IHDX_op2_2), :)), min(D(:, IHDX_op2_2)));

% histo
% figure(100); clf;
% hist(HD1, 50)

OFF = abs(HDD) > 0.5;
vec = [mean(HDx(OFF)), mean(HDy(OFF)), mean(HDz(OFF))];

% XIHD1 = XX();
% 
% HH1 = max(min(D,[],1));
% HH2 = max(min(D,[],2));
% HH = max(HH1,HH2);
return
%% Test
sphere1 = struct('PhantomShape', 'Spherical', 'nBins', 64, 'matSizeCm', 64, 'r', 16/2, 'offset', [0.5,0,0]);
[Ph_image1] = radon_phantom(sphere1);
sphere2 = struct('PhantomShape', 'Spherical', 'nBins', 64, 'matSizeCm', 64, 'r', 16/2, 'offset', [0,0,0]);
[Ph_image2] = radon_phantom(sphere2);
[Hau1, D] = HausdorffDist(Ph_image1, Ph_image2);
Hau1
Hau2 = Hausdorff3D(Ph_image1, Ph_image2)
ibGUI(Ph_image1 + 2*Ph_image2)
