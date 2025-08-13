%% Create phantom, lay out gradients
phi   = -180:360/15:180;
theta = -180:360/15:180;
alpha = 0:1/7:1;
[PHI,THETA] = meshgrid(phi,theta);

% Build gradients for different 
[kx, ky, kz] = sph2cart(PHI(:)*pi/180,THETA(:)*pi/180,.3);
k = [kx,ky,kz]*alpha(1);
for ii=2:length(alpha)
    k = cat(1, k, [kx,ky,kz]*alpha(ii));
end

p_space = (-15.5:15.5)/3;
l1 = (-7.5:7.5)*3;
l2 = (-7.5:7.5)*3;
l3 = (-7.5:7.5)*3;
b1 = (-7.5:7.5)*3;

phantom = zeros(length(l1), length(l2),length(l3),length(b1));
[L1,L2,L3] = meshgrid(l1,l2,l3);
idx = (L1-7).^2+(L2+12).^2+(L3+5).^2 < 6^2;
the_shape = lshape(b1, 0, 7);
[i1,i2,i3] = ind2sub([length(l1), length(l2),length(l3)], find(idx));
for ii=1:length(i1)
    phantom(i1(ii),i2(ii),i3(ii),:)=the_shape;
end
idx = (L1+5).^2+(L2-3).^2++(L3+5).^2 < 6^2;
the_shape = lshape(b1, 0, 16);
[i1,i2,i3] = ind2sub([length(l1), length(l2),length(l3)], find(idx));
for ii=1:length(i1)
    phantom(i1(ii),i2(ii),i3(ii),:)=the_shape;
end

ibGUI(phantom)

nK   = size(k, 1);
nPts = length(p_space);
nObj = length(l1)*length(l2)*length(l3)*length(b1);
fprintf('Obj = %i  <->  nEq = %i\n', nObj, nK*nPts);

%% Generate system matrix
tic
M = epri_SystemMatrix(k, p_space, l1, l2, l3, b1);
toc
size(M)
%% Generate projections

p = reshape(M, [nK*nPts,nObj]);
proj = p*phantom(:);

proj = proj + (rand(size(proj))-0.5)*0.5;
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