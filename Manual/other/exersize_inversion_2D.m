%% Create phantom, lay out gradients
phi = -180:8.4:180;
[kx, ky] = pol2cart(phi'*pi/180,.3);
k = [kx,ky];

p_space = (-21:21)/2;
l1 = (-21:21)/1;
l2 = (-21:21)/1;

phantom = zeros(length(l1), length(l2));
[L1,L2] = meshgrid(l1,l2);
phantom((L1'-7).^2+(L2'+12).^2 < 6^2)=1;

figure(1); clf
subplot(2,2,1);
imagesc(l2,l1,phantom); axis image
xlabel('l1');ylabel('l2')

nK   = size(k, 1);
nPts = length(p_space);
nObj = length(l1)*length(l2);
fprintf('Obj = %i  <->  nEq = %i\n', nObj, nK*nPts);

%% Generate system matrix

M = epri_SystemMatrix(k, p_space, l1, l2, [], []);
size(M)
%% Generate projections

p = reshape(M(:,:,:), [nK*nPts,nObj]);
proj = p*phantom(:);

proj = proj + (rand(size(proj))-0.5)*5;
p1 = reshape(proj, [nK, nPts]);

subplot(2,2,2)
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
rec_im=(LL+50*DD0)\p'*proj;
toc

subplot(2,2,3);
imagesc(l2, l1, reshape(rec_im, [length(l2), length(l1)]))
axis image;
