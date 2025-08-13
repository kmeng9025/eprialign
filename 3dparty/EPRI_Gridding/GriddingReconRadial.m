clear all; close all; clc;
if ismac
    path(path, '/Users/borisepel/Dropbox/MATLAB/3dparty/EPRI_Gridding/common');
    path(path, '/Users/borisepel/Dropbox/MATLAB/3dparty/EPRI_Gridding/common/library');
else
    path(path, 'C:\Users\admin\Dropbox\MATLAB\3dparty\EPRI_Gridding\common');
    path(path, 'C:\Users\admin\Dropbox\MATLAB\3dparty\EPRI_Gridding\common\library');
end

numWorker = 8;  % 1 for single threaded.
scaling = 1.0;  % image scaling factor
bConjSymm = 1;


%% 1. Load and refine input data
load epri_UChi_kdata.mat

proj = yy;
dataIdx = 16:16+256;    %Freq encoded data to be used 
proj = proj(dataIdx,:,:);   % truncate samples
centerIdxYY = 88;   % Center of k-space in yy in dim-1
centerIdx = 88 - dataIdx(1) + 1;   % Center of k-space in proj in dim-1

Nsample = size(proj,1);
numT = size(proj,2);
numProj = size(proj,3);

% Demodulate with center-k
trace = squeeze(proj(:,numT,:));
phase = angle(trace(centerIdx,:));
for i=1:numT
    currProj = squeeze(proj(:,i,:));
    currProj = currProj .* exp( -1i*(ones(Nsample,1) * phase) );
    proj(:,i,:) = currProj;
end

% Estimated unsampled data using conjugate symmetry
if bConjSymm
    NsPadded = 2 * (Nsample - centerIdx) - Nsample;
    tmpProj = zeros(NsPadded+Nsample, numT, numProj);
    tmpProj(NsPadded+1:end,:,:) = proj;
    tmpProj(1:NsPadded,:,:) = ...
        real(proj(end:-1:end-NsPadded+1,:,:)) - 1i*imag(proj(end:-1:end-NsPadded+1,:,:));
    proj = tmpProj;
    Nsample = size(proj,1);
    centerIdx = centerIdx + NsPadded;
end
% figure, imagesc(abs(squeeze(proj(:,1,:))));


%% 1. Load data (NEW)

s1=load('/Users/borisepel/Dropbox/MATLAB/DATA/IRESE2/prj_1892IRESE_0p75Gcm_ox63d24_tau1p8.mat') 
proj = s1.kproj;
Nsample = size(proj,1);
centerIdx = Nsample/2;
numT = size(proj,2);
numProj = size(proj,3);
kx = s1.kxnorm;
ky = s1.kynorm;
kz = s1.kznorm;

%% 2. Set gridding parameter

gridParam.N = 63;
gridParam.zp = 1; % zero padding factor
gridParam.alpha = 4; % grid size
gridParam.gridW = 5; % kernel size
gridParam.gridScale = 33; % kernel resolution
gridParam.dim = 3;
gridParam.nPar = numWorker; % number of thread
Nimg = gridParam.N*gridParam.zp;
NCenter = floor((Nimg+1)/2);


%% 3. Set sampling trajectory
numPts = numProj*Nsample;
traj1d = ( (1:Nsample) - centerIdx );
traj1d = traj1d' / max(traj1d); % Normalized 1D trajectory
traj = zeros(numPts, 3);
for grad = 1:numProj
    trajXs = traj1d * kx(grad);
    trajYs = traj1d * ky(grad);
    trajZs = traj1d * kz(grad);         
    
    tidx1 = (grad-1)*Nsample + 1;
    tidx2 = grad*Nsample;
    
    traj( tidx1:tidx2, : ) = [trajXs(:) trajYs(:) trajZs(:)];
end
% traj = traj / max(abs(traj(:))) * floor(gridParam.N/2);
traj = traj * floor(gridParam.N/2);
gridParam.traj = traj;

%% 4. Do reconstruction
idataGrid = zeros(Nimg,Nimg,Nimg,numT);
for i = 1:numT
    tic
    currProj = squeeze(proj(:,i,:));
    currProj = currProj(:);

    [gridImg, gridK] = DoGriddingNonCarte(currProj(:), gridParam, 1/scaling);
    idataGrid(:,:,:,i) = gridImg;

    timeRecon = toc;    
    disp(sprintf('Recon for echo#%g done : %gs elapsed.', i, timeRecon));
    
    px = NCenter;
    py = NCenter;
    pz = NCenter;  
    
    figure(1), 
    subplot(231); imagesc( abs(getSlice(gridImg,1,px)) ); axis image;
    subplot(232); imagesc( abs(getSlice(gridImg,2,py)) ); axis image;
    title(sprintf('Reconstructed images: echo #%g',i));
    subplot(233); imagesc( abs(getSlice(gridImg,3,pz)) ); axis image;

    subplot(234); imagesc( abs(getSlice(gridK,1,px*2)) ); axis image;
    subplot(235); imagesc( abs(getSlice(gridK,2,py*2)) ); axis image;
    title(sprintf('Gridded k-space: echo #%g',i));    
    subplot(236); imagesc( abs(getSlice(gridK,3,pz*2)) ); axis image;
    drawnow;
end

ibGUI(real(permute(idataGrid, [2,1,3,4])));

% figure, subplot(121); plot(abs(squeeze(idataGrid(20,31,30,:))), 'o')
%         subplot(122); plot( [-abs(squeeze(idataGrid(20,31,30,1:5))); abs(squeeze(idataGrid(20,31,30,6:end)))], 'o')


% 
% % STOPHERE
% % To view results
% for i=1:numT
%     
%     px = NCenter;
%     py = NCenter;
%     pz = NCenter;        
%     
%     figure(1), 
%     subplot(131); imagesc( abs(getSlice(idataGrid,1,px,i)) ); axis image;
%     subplot(132); imagesc( abs(getSlice(idataGrid,2,py,i)) ); axis image; title(i);
%     subplot(133); imagesc( abs(getSlice(idataGrid,3,pz,i)) ); axis image;
%     drawnow;
%     pause;
% end

