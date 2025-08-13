function outmask=outside_mask3(avol,varargin)
% finds the outside mask using largest component for a volume
% threshold defaults to 10% if not passed.

if isempty(varargin)
    thresh=0.1;
else
    thresh=varargin{1};
end
outmask=largest_component(avol,thresh);
nslices=size(outmask,3);
for n=1:nslices
  mask1=squeeze(outmask(:,:,n));
  mask2=imfill(mask1,[1 1]);
  outmask(:,:,n)=mask1+(1.0-mask2);
end
return