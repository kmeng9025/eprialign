% function Spec=bin2D(Spec,x1,x2,k1,k2,amp)
% binning to the array Spec
% x1,x2, spec axis
% k1,k2,amp bin position and amplitude

function Spec=bin2D(Spec,x1,x2,k1,k2,amp)

sz = size(Spec);
mmax1 = max(x1) - min(x1);
mmax2 = max(x2) - min(x2);
k1_idx = round((k1-min(x1))/mmax1*length(x1));
k2_idx = round((k2-min(x2))/mmax2*length(x2));
idx = k1_idx >0 & k2_idx > 0 & k1_idx<=sz(1) & k2_idx<=sz(2);

iidx = sub2ind(sz, k1_idx(idx), k2_idx(idx));
Spec(iidx) = Spec(iidx) + amp(idx);
