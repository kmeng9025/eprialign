function mat_XYZB = bp4d_multistage(mat, lagOffset, pinterpflag, ifactor, imtype)
%
% Last modified by KA dd061026
% recon_gui version 0.1

% lagOffset = lagshift*nbins/orgnbins
dataTypeStr=class(mat);
[RhoBPhiTheta, RZPhiB, x, y, ctrPt, num, expWt] = CreateWorkSpace(imtype, mat);
[cosalpha, sinalpha, costheta, sintheta, cosphi, sinphi] = ...
    ComputeTrigonometry(num, pinterpflag, ifactor);

time0=cputime;
% Spectral BackProjection
% XiAlphaPhiTheta(=mat) => RhoBPhiTheta

% precalculate the interpolation index arrays
%  outside the for j and for k loops

if num.Spec > 1
    [a, delta]=ComputeIndexSet(cosalpha,sinalpha,x,y,ctrPt+lagOffset);
    fprintf('Entering first loop...');
    for j=1:num.Az, for k=1:num.Polar,
        m2D=zeros(num.R,num.R,dataTypeStr);
        for i=1:num.Spec
            proj=[0 mat(:,i,j,k)' 0 0]';
            wt = pi*abs(sinalpha(i))^(expWt.Polar+expWt.Az);
            m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*wt;
        end
        RhoBPhiTheta(:,:,j,k)=m2D/num.Spec;
    end, end
else
    RhoBPhiTheta=mat;
end % END OF if nSpec > 1
    time1=cputime;
    fprintf('Done.\n cpu time for loop 1: %5.3f s\n', time1-time0);
    clear mat;

% Polar BackProjection
% RhoBPhiTheta => RZPhiB
if num.Polar > 1
    [a, delta]=ComputeIndexSet(costheta,sintheta,x,y,ctrPt);
    fprintf('second loop...');
    for j=1:num.B, for k=1:num.Az,
        m2D=zeros(num.R,num.R,dataTypeStr);
        for i=1:num.Polar
            proj=[0 RhoBPhiTheta(:,j,k,i)' 0 0]';
            wt=pi*abs(sintheta(i))^expWt.Az;
            m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*wt;
        end
        RZPhiB(:,:,k,j)=m2D/num.Polar;
    end, end
else
    for k=1:num.Az
        RZPhiB(:,1,k,:)=RhoBPhiTheta(:,:,k,1);
    end
end % END OF if nPolar > 1 
    time2=cputime;
    fprintf('Done.\n cpu time for loop 2: %5.3f s\n', time2-time1);
    clear RhoBPhiTheta;

mat_XYZB=zeros(num.X,num.Y,num.Z,num.B,dataTypeStr);
if num.Az > 1
    [a, delta]=ComputeIndexSet(cosphi,sinphi,y,x,ctrPt);
    fprintf('third loop...');
    for j=1:num.Z, for k=1:num.B,
        m2D=zeros(num.R,num.R,dataTypeStr);
        for i=1:num.Az
            proj=[0 RZPhiB(:,j,i,k)' 0 0]';
            m2D(:)=m2D(:)+(delta(:,i).*proj(a(:,i)+1) + (1-delta(:,i)).*proj(a(:,i)))*pi;
        end
        mat_XYZB(:,:,j,k)=m2D/num.Az;
    end, end
else
    mat_XYZB=RZPhiB;
end % END OF if nAz > 1
time3=cputime;
fprintf('Done.\n cpu time for loop 3: %5.3f s\n', time3-time2);
fprintf('Done.\n cpu time for Back Projection: %5.3f s\n', time3-time0);
% mat_XYZB=mat_XYZB/pi^3;   %%%%%%%%%%%%%%%  What is the reason to adjust
% the value; IT IS A BUG FOUND BY ZHIWEI 

function [cosalpha, sinalpha, costheta, sintheta, cosphi, sinphi]=...
    ComputeTrigonometry(num, pinterpflag, ifactor)
% sinc interpolation (pinterpflag=1) results in an interpolated
%	sinogram that starts with the initial angle = pi/2 - pi/N/2
% local interpolation (pinterpflag > 1) results in an interpolated
%	sinogram that starts with the initial angle = pi/2 - pi/Ni/2
% N = number of angles before interpolation
% Ni = number of angles after interpolation (Ni=N*ifactor)
% dd051118 KA
if pinterpflag>1 && ifactor > 1
    isSinc=0;
else	% "sinc interpolation" defines same initial angle as "no interpolation" 
    isSinc=1;
end

d = pi/num.Spec; a = 0.5*pi-0.5*d*ifactor^isSinc;
alpha = a - d*(0:num.Spec-1);
cosalpha=cos(alpha);
sinalpha=sin(alpha);

d = pi/num.Polar; a = 0.5*pi-0.5*d*ifactor^isSinc;
theta = a - d*(0:num.Polar-1);
costheta=cos(theta);
sintheta=sin(theta);

d = pi/num.Az; a = 0.5*pi-0.5*d*ifactor^isSinc;
phi = a - d*(0:num.Az-1);
cosphi=cos(phi);
sinphi=sin(phi);
% end of function ComputeTrigonometry

function [RhoBPhiTheta, RZPhiB, x, y, ctrPt, num, expWt] = CreateWorkSpace(imtype, mat)
dataTypeStr=class(mat);
num.R=size(mat,1);
if dataTypeStr=='single', num.R=single(num.R);end
expWt.Polar=0; expWt.Az=0;
num.Polar=size(mat,4);
num.Az=size(mat,3);
if num.Polar > 1, expWt.Polar=1;end
if num.Az > 1, expWt.Az=1;end
num.Spec=size(mat,2);
[num.X,num.Y,num.Z,num.B]=get_image_dims(imtype,num.R);
RhoBPhiTheta=zeros(num.R,num.B,num.Az,num.Polar,dataTypeStr);
RZPhiB=zeros(num.R,num.Z,num.Az,num.B,dataTypeStr);
ctrPt = (num.R+1)/2;
rVec=-ctrPt+(1:num.R);
[x, y] = meshgrid(rVec,rVec);
% end of function CreateWorkSpace

function [a, delta] = ComputeIndexSet(cosA, sinA, x, y, ctrPt)
N=size(x,1);
K=length(cosA);
dataTypeStr=class(ctrPt);
delta=zeros(N*N,K,dataTypeStr);
a=zeros(N*N,K,dataTypeStr);
t=zeros(N*N,K,dataTypeStr);

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