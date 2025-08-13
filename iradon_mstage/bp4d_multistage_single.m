function mat_XYZB = bp4d_multistage_single(mat, lagOffset, pinterpflag, ifactor, imtype)
% lagOffset = lagshift*nbins/orgnbins
nPolar=size(mat,4);
nAz=size(mat,3);
nSpec=size(mat,2);
nbins=size(mat,1);
 
[RhoBPhiTheta, RZPhiB, x, y, ctrPt] = CreateWorkSpace(nbins, nAz, nPolar);
[cosalpha, sinalpha, costheta, sintheta, cosphi, sinphi] = ...
    ComputeTrigonometry(nSpec, nPolar, nAz, pinterpflag, ifactor);

time0=cputime;
% Spectral BackProjection
% XiAlphaPhiTheta(=mat) => RhoBPhiTheta

% precalculate the interpolation index arrays
%  outside the for j and for k loops
[a, delta]=ComputeIndexSet(cosalpha,sinalpha,x,y,ctrPt+lagOffset);
fprintf('Entering first loop...');
for j=1:nAz, for k=1:nPolar,
        m2D=single(0); m2D(nbins,nbins)=0;
for i=1:nSpec
    proj=[0 mat(:,i,j,k)' 0 0]';
    m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*sinalpha(i)^2;
end
RhoBPhiTheta(:,:,j,k)=m2D/nSpec;
end, end
time1=cputime;
fprintf('Done.\n cpu time for loop 1: %5.3f s\n', time1-time0);
clear mat; pack;

% Polar BackProjection
% RhoBPhiTheta => RZPhiB

[a, delta]=ComputeIndexSet(costheta,sintheta,x,y,ctrPt);
fprintf('second loop...');
for j=1:nbins, for k=1:nAz,
        m2D=single(0); m2D(nbins,nbins)=0;
for i=1:nPolar
    proj=[0 RhoBPhiTheta(:,j,k,i)' 0 0]';
    m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*abs(sintheta(i));
end
RZPhiB(:,:,k,j)=m2D/nPolar;
end, end
time2=cputime;
fprintf('Done.\n cpu time for loop 2: %5.3f s\n', time2-time1);
clear RhoBPhiTheta; pack;
mat_XYZB=single(0);
mat_XYZB(nbins,nbins,nbins,nbins)=0;

[a, delta]=ComputeIndexSet(cosphi,sinphi,y,x,ctrPt);
fprintf('third loop...');
for j=1:nbins, for k=1:nbins,
        m2D=single(0); m2D(nbins,nbins)=0;
for i=1:nAz
    proj=[0 RZPhiB(:,j,i,k)' 0 0]';
    m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)));
end
mat_XYZB(:,:,j,k)=m2D/nAz;
end, end
time3=cputime;
fprintf('Done.\n cpu time for loop 3: %5.3f s\n', time3-time2);
fprintf('Done.\n cpu time for Back Projection: %5.3f s\n', time3-time0);

function [cosalpha, sinalpha, costheta, sintheta, cosphi, sinphi]=...
    ComputeTrigonometry(nSpec, nPolar, nAz, pinterpflag, ifactor)
% sinc interpolation (pinterpflag=1) results in an interpolated
%	sinogram that starts with the initial angle = pi/2 - pi/N/2
%	(N=number of sample before interpolation)
% spline interpolation (pinterpflag=2) results in an interpolated
%	sinogram that starts with the initial angle = pi/2
% dd050929 KA
if pinterpflag==2 && ifactor > 1
    isSinc=0;
else	% "sinc interpolation" defines same initial angle as "no interpolation" 
    isSinc=1;
end

d = pi/nSpec; a = 0.5*pi-0.5*d*ifactor*isSinc;
alpha = a - d*(0:nSpec-1);
cosalpha=single(cos(alpha));
sinalpha=single(sin(alpha));

d = pi/nPolar; a = 0.5*pi-0.5*d*ifactor*isSinc;
theta = a - d*(0:nPolar-1);
costheta=single(cos(theta));
sintheta=single(sin(theta));

d = pi/nAz; a = 0.5*pi-0.5*d*ifactor*isSinc;
phi = a - d*(0:nAz-1);
cosphi=single(cos(phi));
sinphi=single(sin(phi));
% end of function ComputeTrigonometry

function [RhoBPhiTheta, RZPhiB, x, y, ctrPt] = CreateWorkSpace(nbins,nAz,nPolar)
RhoBPhiTheta=single(0);
RhoBPhiTheta(nbins,nbins,nAz,nPolar)=0;
RZPhiB=single(0);
RZPhiB(nbins,nbins,nAz,nbins)=0;
ctrPt = single((nbins+1)/2); 
rVec=-ctrPt+(1:nbins);
[x, y] = meshgrid(rVec,rVec);
% end of function CreateWorkSpace

function [a, delta] = ComputeIndexSet(cosA, sinA, x, y, ctrPt)
N=size(x,1);
K=length(cosA);
delta=single(0);a=single(0);t=single(0);
delta(N*N,K)=0;a(N*N,K)=0;t(N*N,K)=0;

for k=1:K
    tp=x*cosA(k)+y*sinA(k);
    tp=tp(:)+ctrPt+1;
    tp(find(tp>N+1.5))=N+2;
    tp(find(tp<1.5))=1;
    a(:,k)=floor(tp);
    t(:,k)=tp;
end
    delta=t-a;
% end of function ComputeIndexSet