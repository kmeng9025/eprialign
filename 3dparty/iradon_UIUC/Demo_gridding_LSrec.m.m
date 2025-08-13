% Demo 1: Gridding-based least squres reconstruction
clear all, close all
clc

addpath('hftrec')
addpath('gridding_ls_3D')

%% Load data and set up coordinates
load('TCD50_MCa4_M61_002.mat');
mat = mat - mat_bl;  %correct baseline

%% Actual DC (asymmetric echoes)
DC = rec_info.td.max_idx;
first_idx = [];
for ii=1:size(mat,2)
  idx = raw_info.t_ax+rec_info.td.acq_window_start*1E-9+raw_info.tau2(ii)/2 <  rec_info.td.dead_time*1E-9;
  kkk = find(idx,1,'last');
  if ~isempty(kkk)
    first_idx(ii) = kkk;
  else
    first_idx(ii) = 1;
  end
end
clear idx kkk
Nread = 2*(size(mat,1)-DC);
mat_old = mat;
mat = complex(zeros(Nread,size(mat,2),size(mat,3)));
for ii=1:size(mat,2)
  mat(:,ii,:)=fftshift(fft(fftshift(hftrec(mat_old(first_idx(ii):end,ii,:),Nread,DC-first_idx(ii)+1),1)),1);
end
clear mat_old
readout = linspace(-1,1,Nread);

%% Setup k-space coordinates
mtxsize = rec_info.rec.Sub_points;
delta_t = mean(diff(raw_info.t_ax));

kx = readout.' * raw_info.GradX.' * mtxsize/2 * sqrt(2);
ky = readout.' * raw_info.GradY.' * mtxsize/2 * sqrt(2);
kz = readout.' * raw_info.GradZ.' * mtxsize/2 * sqrt(2);

% % Choose k-space samples
k_space = permute(mat, [1 3 2]);
k_space = reshape(k_space, [], size(k_space,3));

%% Nonuniform Fourier-based least-squares reconstruction
disp(' ')
disp('Least Squares:')
Nx = mtxsize;
Ny = mtxsize;
Nz = mtxsize;

b = zeros(Nx*Ny*Nz, size(k_space,2));
for ii=1:size(k_space,2)
     b(:, ii) = Fuh_nonCart_3D(k_space(:, ii), kx(:), ky(:), kz(:), Nx, Ny, Nz);
end

recon_ls = complex(zeros([Nx, Ny, Nz, size(k_space,2)]));
Qf = Qmat_3D(Nx, Ny, Nz, kx(:), ky(:), kz(:), size(k_space, 1));

tol = 1e-5;
maxit = 100;

for ii = 1:size(k_space, 2)
  [Irec, flag, relres, iter, resvec] = pcg(@(x) (FhF3D(x, Qf)), b(:, ii), tol, maxit);
  recon_ls(:, :, :, ii) = reshape(Irec, Nx, Ny, Nz);
  % Display the reconstructed 3D volume at each TR
  figure, imshow(imagesc3d(abs(recon_ls(:, :, :, ii))), [])
  title('Least Squares'), pause(1);
end

