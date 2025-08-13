%% Create phantom, lay out gradients
phi   = -180:15:180;
theta = -180:15:180;
[PHI,THETA] = meshgrid(phi,theta);

[kx, ky, kz] = sph2cart(PHI(:)*pi/180,THETA(:)*pi/180,.3);
k = [kx,ky,kz];

p_space = (-12:12);
l1 = (-12:12)*1.6;
l2 = (-12:12)*1.6;
l3 = (-12:12)*1.6;

phantom = zeros(length(l1), length(l2),length(l3));
[L1,L2,L3] = meshgrid(l1,l2,l3);
phantom((L1-7).^2+(L2+12).^2++(L3+5).^2 < 6^2)=1;
phantom((L1+5).^2+(L2-3).^2++(L3+5).^2 < 6^2)=1;

ibGUI(phantom)

nK   = size(k, 1);
nPts = length(p_space);
nObj = length(l1)*length(l2)*length(l3);
fprintf('Obj = %i  <->  nEq = %i\n', nObj, nK*nPts);

%% Generate system matrix
tic
M = epri_SystemMatrix(k, p_space, l1, l2, l3, []);
toc
size(M)
%% Generate projections

p = reshape(M(:,:,:), [nK*nPts,nObj]);
proj = p*phantom(:);

proj = proj + (rand(size(proj))-0.5)*5;
p1 = reshape(proj, [nK, nPts]);

figure(1);
imagesc(p_space, phi, p1);
axis square;
xlabel('pspace');ylabel('phi')

%%
tic
% Moore-Penrose
% rec_im = pinv(p, 5)*proj;

% Tikhonov
v0=ones(1,nK*nPts);
D0=diag(v0,0);
DD0=D0'*D0;
LL=p'*p;
rec_im=(LL+5*DD0)\p'*proj;
toc

ibGUI(reshape(rec_im, [length(l1), length(l2), length(l3)]))