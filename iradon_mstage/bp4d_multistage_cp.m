function mat_XYZB = bp4d_multistage_cp(mat, lagOffset, pinterpflag, ifactor, imtype);
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

% precalculate the interpolation index arrays outside the for j and for k
% loops
t=zeros(nbins*nbins,max([nSpec nPolar nAz]));
for i=1:nSpec
    t(:,i) = ComputeIndexSet1(cosalpha(i), sinalpha(i), x, y, nbins, ctrPt+lagOffset);
end
a=floor(t);
delta=t-a;
    
fprintf('Entering first loop...');
for j=1:nAz, for k=1:nPolar,
m2D=zeros(nbins,nbins);
for i=1:nSpec
    proj=[0 mat(:,i,j,k)' 0 0]';
    %[a, t] = ComputeIndexSet(cosalpha(i), sinalpha(i), x, y, nbins, ctrPt+lagOffset);
    %m2D=m2D+((t-a).*proj(a+1) + (a+1-t).*proj(a))*sinalpha(i)^2;
    m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*sinalpha(i)^2;
end
RhoBPhiTheta(:,:,j,k)=m2D/nSpec;
end, end
time1=cputime;
fprintf('Done.\n cpu time for loop 1: %5.3f s\n', time1-time0);
clear mat; pack;

% Polar BackProjection
% RhoBPhiTheta => RZPhiB

%t=zeros(nbins*nbins,nPolar);
for i=1:nPolar
    t(:,i) = ComputeIndexSet1(costheta(i), sintheta(i), x, y, nbins, ctrPt);
end
a=floor(t);
delta=t-a;

fprintf('second loop...');
for j=1:nbins, for k=1:nAz,
m2D=zeros(nbins,nbins);
for i=1:nPolar
    proj=[0 RhoBPhiTheta(:,j,k,i)' 0 0]';
%    [a, t] = ComputeIndexSet(costheta(i), sintheta(i), x, y, nbins, ctrPt);
 %   m2D=m2D+((t-a).*proj(a+1) + (a+1-t).*proj(a))*abs(sintheta(i));
     m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*abs(sintheta(i));
end
RZPhiB(:,:,k,j)=m2D/nPolar;
end, end
time2=cputime;
fprintf('Done.\n cpu time for loop 2: %5.3f s\n', time2-time1);
clear RhoBPhiTheta; pack;
mat_XYZB=zeros(nbins,nbins,nbins,nbins);
%a=1:size(RZPhiB,1)+3;
% Azimuthal BackProjection
% RZPhiB => XYZB
fprintf('third loop...');

%t=zeros(nbins*nbins,nAz);
for i=1:nAz
    t(:,i) = ComputeIndexSet1(cosphi(i), sinphi(i), y, x, nbins, ctrPt);
end
a=floor(t);
delta=t-a;

for j=1:nbins, for k=1:nbins,
m2D=zeros(nbins,nbins);
for i=1:nAz
    proj=[0 RZPhiB(:,j,i,k)' 0 0]';
%    [a,t] = ComputeIndexSet(cosphi(i), sinphi(i), y, x, nbins, ctrPt);
    m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)));
end
mat_XYZB(:,:,j,k)=m2D/nAz;
end, end
time3=cputime;
fprintf('Done.\n cpu time for loop 3: %5.3f s\n', time3-time2);
%time0=cputime-time0;
fprintf('Done.\n cpu time for Back Projection: %5.3f s\n', time3-time0);

function [cosalpha, sinalpha, costheta, sintheta, cosphi, sinphi]=...
    ComputeTrigonometry(nSpec, nPolar, nAz, pinterpflag, ifactor);
% sinc interpolation (pinterpflag=1) results in an interpolated
%	sinogram that starts with the initial angle = pi/2 - pi/N/2
%	(N=number of sample before interpolation)
% spline interpolation (pinterpflag=2) results in an interpolated
%	sinogram that starts with the initial angle = pi/2
% dd050929 KA
if pinterpflag==2 & ifactor > 1
    isSinc=0;
else	% "sinc interpolation" defines same initial angle as "no interpolation" 
    isSinc=1;
end

d = pi/nSpec; a = 0.5*pi-0.5*d*ifactor*isSinc;
alpha = a - d*(0:nSpec-1);
cosalpha=cos(alpha);
sinalpha=sin(alpha);

d = pi/nPolar; a = 0.5*pi-0.5*d*ifactor*isSinc;
theta = a - d*(0:nPolar-1);
costheta=cos(theta);
sintheta=sin(theta);

d = pi/nAz; a = 0.5*pi-0.5*d*ifactor*isSinc;
phi = a - d*(0:nAz-1);
cosphi=cos(phi);
sinphi=sin(phi);
% end of function ComputeTrigonometry

function [RhoBPhiTheta, RZPhiB, x, y, ctrPt] = CreateWorkSpace(nbins,nAz,nPolar);
RhoBPhiTheta=zeros(nbins,nbins,nAz,nPolar);
RZPhiB=zeros(nbins,nbins,nAz,nbins);
ctrPt = (nbins+1)/2; 
rVec=-ctrPt+(1:nbins);
[x, y] = meshgrid(rVec,rVec);
% end of function CreateWorkSpace
function [a,t] = ComputeIndexSet(cosA, sinA, x, y, nbins, ctrPt);
    t=x*cosA+y*sinA;
    t=t+ctrPt+1;
    t(find(t>nbins+1.5))=nbins+2;
    t(find(t<1.5))=1;
    a=floor(t);

function [t] = ComputeIndexSet1(cosA, sinA, x, y, nbins, ctrPt);
    tp=x*cosA+y*sinA;
    t=tp(:)+ctrPt+1;
    t(find(t>nbins+1.5))=nbins+2;
    t(find(t<1.5))=1;
    %a=floor(t);
% end of function ComputeIndexSet