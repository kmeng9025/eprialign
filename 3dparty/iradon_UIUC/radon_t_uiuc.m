function pr=radon_t_uiuc(I, radon_pars)

phi=radon_pars.phi*180/pi;
theta=radon_pars.theta*180/pi;

projections = numel(radon_pars.z);
mtx_size=size(I,1);
if ~( ( size(I,1) == size(I,2) ) && ( size(I,1) == size(I,3) ) )
  disp('Warning: matrix is not a cube. Assuming matrix size from first dimension');
end

pr=zeros(mtx_size,projections);

newpool=isempty(gcp('nocreate'));
if newpool
pool=parpool(6);
end
parfor proj=1:projections
  I_rot = imrotate(permute(I(:,:,:,proj),[2 1 3 4]),-phi(proj),'bilinear','crop');
  I_rot = ipermute(imrotate(permute(I_rot,[3 2 1 4]),-theta(proj),'bilinear','crop'),[3 2 1 4]);
  pr(:,proj)=sum(sum(I_rot,1),3);
end
if newpool
delete(pool);
end
pr=flipud(pr);

return