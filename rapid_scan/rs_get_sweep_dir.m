function needs_flip = rs_get_sweep_dir(y)

nPrj = size(y, 2);
needs_flip = zeros(nPrj, 1);
for ii=1:nPrj
  % split in halves
  % nPts = length(y);
  % isflip = get_flip(y(1:fix(nPts/2))) | get_flip(flipud(y(fix(nPts/2)+1:nPts)));
  needs_flip(ii) = get_flip(y(:,ii));
end

function isflip = get_flip(y)
% routh estimation of maximum
[~, imax] = max(abs(y));
imy = imag(y); imy(imy < max(abs(y))* .25) = 0;
[~, idmax] = max(imy);
[~, idmin] = min(imy);
isflip = idmax > imax && imax > idmin;
