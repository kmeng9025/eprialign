% IRADON_D2D_4D_DIRECT  reconstuct a 3D or 4D object from 1D projections
% [object] = iradon_d2d_mstage(p, radon_pars, recon_pars);
% Algorithm: multi_stage backprojection method
% Projeciton sampling method: equal angle
% Number of bins in reconstructed matrix in all dimensions is equal
% to the number of points in projections
% p - [array, 4D] of projections
%     for 3D - size(p)=[points_in_projection, 1, nTheta, nPhi]
%     for 4D - size(p)=[points_in_projection, nTheta, nPhi, nAlpha]
% radon_pars - [structure] projection parameters
%     [].ELA - [structure] of equal angle gradient scheme parameters
%            [].imtype - Image type [int, 1 for 4D, 14 for 3D]
%            [].nPolar - Number of polar angles [int]
%            [].nAz    - Number of azimuthal angles [int]
%            [].nSpec  - Number of spectral angles[int]
%     [].size - length of the spacial projection [float, in cm]
% recon_pars - [structure] reconstruction parameters
%     [].nBins  - Image size in voxels [int]
%     [].Filter - [string, ram_lak/shepp-logan/cosine/hamming/hann]
%     [].FilterCutOff  - Filter cut off, part of full bandwidth [float, 0 to 1]
%     [].InterpFactor  - Projection interpolation factor, [int, 1/2/4/etc]
%     [].Interpolation - Inerpolation method, [string, (none)/sinc/spline/linear]
%     [].CodeFlag      - Reconstruction code [string, C/MATLAB/FORTRAN]
%     [].zeropadding   - zeropadding factor [int, >= 1]
% object - reconstructed object

% Author: Mark Tseylin, interface Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2014
% Contact: epri.uchicago.edu

function [object, recon_pars, other]=iradon_d2d_direct(data_in, raw_info, recon_pars)

other = [];
dispstat('','init'); % One time only initialization

fbp_struct = raw_info.data.FBP;
ppr_struct = recon_pars.ppr;
prc_struct = recon_pars.prc;
[pars, pars_ext] = iradon_FBPGradTable(fbp_struct); % gradient in pars


% number of harmonics
MNH = safeget(recon_pars.rec, 'nHarmonics', 80);
nBins   = safeget(recon_pars.rec, 'nBins', 12);
if numel(nBins) == 1, nBins = nBins * ones(3,1); end
nBinsField   = safeget(recon_pars.rec, 'nBinsField', 1000);
L   = safeget(recon_pars.rec, 'size', 2.5);
CenterXYZ = safeget(recon_pars.rec, 'center', [0,0,0]);

dx=L/(nBins(1)-1); x=(-L/2:dx:L/2)+CenterXYZ(1);
dy=L/(nBins(2)-1); y=(-L/2:dy:L/2)+CenterXYZ(2);
dz=L/(nBins(3)-1); z=(-L/2:dz:L/2)+CenterXYZ(3);
N3=length(x)*length(y)*length(z);     

G2 = safeget(data_in, 'G', []);

nP=length(data_in.P);             % Number of projections

h=data_in.B{1};
nB=length(h);             % # of field points

% rearrange projections into an array
PR = zeros(nP, nB);
for k=1:nP
    PR(k,:)=zeroLine(data_in.P{k},.05);
end

% 2D matrix of voxel coordinates
[mx,my,mz] = meshgrid(x,y,z);

RR = L/2;
idx = (mx-CenterXYZ(1)).^2+mz.^2 <= RR^2; % cyllinder
X2 = [mx(idx),my(idx),mz(idx)];
nVox = size(X2,1);

fprintf('Voxels: overall %i used %i.\n', N3, nVox)

% vector r(xyz) & vector G Matrix
GR=X2*G2'; % shifts in the  B-domain (r,G)
    
% Reconstruction 
% tol_pen=1e1;
% method='pen-rose';

tol_tikh = safeget(recon_pars.rec, 'tolerance', 200);
method='tikh_0';

disp(sprintf('Method used: %s.', method));

% obtain frequencies
[v d]=fftM(h,h);
w=2*pi*fftshift(v);    % ifft?
wi=-w(1:MNH); % take only MNH harmonics

tic

dispstat(sprintf('Building system matrix.'),'timestamp')

% Build system matrices for every harmonic
TT=zeros(nP,nVox,MNH);
for n=1:nP
    shifts=GR(:,n); 
    [W,SH] = meshgrid(wi,shifts);          % w= DC ... wmax/w -wmax/2 ... -dw.
    T=exp(+1i*SH.*W);  
    TT(n,:,:)=T; % f-domain shift matrix
end

tstamp_1 = toc;
dispstat(sprintf('System matrix. Elapsed: %5.3g s.', tstamp_1),'timestamp')

% FT of projection data
PRw=fft(PR,[],2); %

% Matrix inversion
fH=1; % first harmonic
x_PH=zeros(nVox,nB);
switch method
    case 'pen-rose'
        for m=fH:MNH %length(w)
            L=TT(:,:,m);
            b=PRw(:,m);
            xx=pinv(L,tol_pen)*b;
            x_PH(:,m)=xx;
            disp(m)
        end
    case 'tikh_0'
        % Regularization operators
        v0=ones(1,nVox);
        D0=diag(v0,0);
        DD0=D0'*D0;
        % Solve 3D problem for each harmonic
        for m=fH:MNH %length(w)
            L=TT(:,:,m);
            b=PRw(:,m);
            LL=L'*L;
            xx=(LL+tol_tikh*DD0)\(L'*b);
            x_PH(:,m)=xx;
            tstamp_2 = toc;
            dispstat(sprintf('Harmonic: %i (%i). Elapsed: %5.3g s. Left: %5.3g s.', m, MNH, tstamp_2, (tstamp_2-tstamp_1)*(MNH-m)/m),'timestamp')
        end 
end

dispstat(sprintf('Finished. Elapsed: %5.3g s.', toc), 'keepthis');

x_Ph=real(ifft(x_PH,[],2));
% x_Ph=x_Ph/max(x_Ph(:));

% Ph_rec=x_Ph;
% for ii=1:N3
%     Ph_rec(ii,:)=zeroLine(x_Ph(ii,:),0.05);
% end

for ii=1:nVox
  interpPh_rec(ii, :) = interp1(1:nB, x_Ph(ii, :), linspace(1,nB, nBinsField));
end

object = zeros(N3, nBinsField);
object(idx,:) = interpPh_rec;
object = reshape(object, [nBins(1), nBins(2), nBins(3), nBinsField]);
object = permute(object, [2,1,3,4]);

