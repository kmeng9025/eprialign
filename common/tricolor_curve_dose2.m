function dose = tricolor_curve_dose2( rgb,r,g,b,d,varargin )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% pare down the problem by bracketing the minimum:  find the dose predicted
% by the red value, and go one interval above and below it on the calibration
% curve.
rtest=double(rgb(1));
rdose=interp1(r,d,rtest,'linear','extrap');
if d(end) > d(1),  %cal curve in increasing dose order
    if (d(end) < rdose) dose=d(end); return;
    else
        idose=min(find(d>=rdose));
    end
    ihi=min([numel(d) idose+1]);
    ilo=max([1 idose-2]);
else  % cal curve in decreasing dose order
    if (d(1) < rdose) dose=d(1); return
    else
        idose=max(find(d>=rdose));
    end
    ihi=min([numel(d) idose+2]);
    ilo=max([1 idose-1]);
end


nsegs=size(r,1);
segs=zeros(nsegs,3);
for n=1:nsegs
    segs(n,:)=double([r(n) g(n) b(n)]);
end
if(size(varargin,2) == 2),
    uvec1=varargin{1};
    len1=varargin{2};
    [dists,closest,alphas] = point_to_poly_1(double(rgb),segs(ilo:ihi,:),uvec1(ilo:ihi-1,:),len1(ilo:ihi-1,:));
else
[dists,closest,alphas] = point_to_poly_1(double(rgb),segs(ilo:ihi,:));end
goodalph=find(alphas>=0 & alphas <= 1.0);
if (~isempty(goodalph))

    if (numel(goodalph)>1),; 
        whichalph=min(find(dists(goodalph)==min(dists(goodalph))));
        goodalph=goodalph(whichalph);
    end
    dose=d(ilo-1+goodalph)+alphas(goodalph)*(d(ilo-1+goodalph+1)-d(ilo-1+goodalph));
else
    dists=zeros(ihi-ilo+1,1);
    for n=ilo:ihi
        fromto=segs(n,:)-double(rgb(1,:));
        dists(n+1-ilo)=dot(fromto,fromto);
    end
    
    theone = find(dists == min(dists));
    dose = d(ilo-1+theone);
end

