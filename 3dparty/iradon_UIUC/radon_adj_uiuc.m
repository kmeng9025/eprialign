function I=radon_adj_uiuc(pr, radon_pars)

phi=radon_pars.phi*180/pi;
theta=radon_pars.theta*180/pi;

projections = size(pr,2);
mtx_size=size(pr,1);

I=zeros(mtx_size,mtx_size,mtx_size);

pr=flipud(pr);
newpool=isempty(gcp('nocreate'));
if newpool
pool=parpool(6);
end
parfor proj=1:projections
  backproj=repmat(pr(:,proj).',[mtx_size 1 mtx_size]);
  backproj=ipermute(imrotate(permute(backproj,[3 2 1]),theta(proj),'bilinear','crop'),[3 2 1]);
  I=I+imrotate(backproj,phi(proj),'bilinear','crop');
end
if newpool
delete(pool);
end
I=ipermute(I,[2 1 3]);

return