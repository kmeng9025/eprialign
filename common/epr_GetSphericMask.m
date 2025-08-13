% Mask = epr_GetSphericMask(Dim, center, rad)
function [Mask, dist_mastrix] = epr_GetSphericMask(Dim, center, rad)

d1 = 1:Dim(1); d1 = repmat(d1', [1, Dim(2), Dim(3)]);
d2 = 1:Dim(2); d2 = repmat(d2, [Dim(1), 1, Dim(3)]);
d3 = 1:Dim(3); d3 = repmat(reshape(d3, [1,1,Dim(3)]), [Dim(1), Dim(2), 1]);

R2 = (d1 - center(1)).^2 + (d2 - center(2)).^2 + (d3 - center(3)).^2;

Mask = false(Dim);
if rad == 0
    pos = round(center);
    Mask(pos(1), pos(2), pos(3)) = 1;
else
    Mask(R2 <= rad^2) = true;
end

if nargout == 2
    dist_mastrix = sqrt(R2);
end
