% IRADON_UIUC_LS  reconstuct a 3D object from 1D projections
% [object] = iradon_uiuc_ls(p, radon_pars, recon_pars);
% Algorithm: see inside /iradon_UIUC directory
% p - [array, 2D] of projections in spatial domain
%     for 3D - size(p)=[points_in_projection, n_projections]
% radon_pars - [structure] projection parameters
%     [].x - Projection unit vector component, [array, 1D] 
%     [].y - Projection unit vector component, [array, 1D] 
%     [].z - Projection unit vector component, [array, 1D] 
%     [].size - Spatial extent of projection [float, in cm]
% recon_pars - [structure] reconstruction parameters
%     [].nBins  - Image size in voxels [int]
%     [].size   - Image size [float, in cm]
% object - reconstructed object

% Author: Boris Epel
% For authors of reconstruction see inside /iradon_UIUC directory
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function object=iradon_uiuc_ls(p, radon_pars, recon_pars)

zeropadding = safeget(recon_pars, 'zeropadding', 1);

% make k-space data
[p, zeropadding]=iradon_zeropadding(p,zeropadding);
k_space = fftshift(fft(fftshift(p), [], 1));

nP = size(k_space, 1);
dk =1/radon_pars.size/zeropadding;

% k in [cm-1]
kx=(-nP/2:nP/2-1)'*radon_pars.x(:)'*dk; 
ky=(-nP/2:nP/2-1)'*radon_pars.y(:)'*dk;
kz=(-nP/2:nP/2-1)'*radon_pars.z(:)'*dk;

% k in dx's - normalized
kx=kx*recon_pars.size;  
ky=ky*recon_pars.size;
kz=kz*recon_pars.size;

maxit = 100;
tol = 1e-5;
nBins=recon_pars.nBins;% the number of the points in the image

disp('UIUC LS reconstruction');
tic
Fhd = Fuh_nonCart_3D(k_space(:), kx(:), ky(:), kz(:), nBins, nBins, nBins); %  the first step  
t1=toc;
fprintf('Fhd time %f seconds \n', t1);

tic
Qf = Qmat_3D(nBins, nBins, nBins, kx(:), ky(:), kz(:), length(k_space(:)));%  the second step
t2=toc;
fprintf('Qf time %f seconds \n', t2);

tic
[object, flag, relres, iter, resvec] = pcg(@(x) (FhF3D(x, Qf)), Fhd(:), tol, maxit);% the third step
t3=toc;
fprintf('Object computing time %f seconds \n', t3);

fprintf('Total time %f seconds\n', t1+t2+t3);

object=reshape(object,[ nBins, nBins, nBins]);

object=real(object)*(nBins/32)^1.5/pi; 