function [mat_int, nplr, naz, nspec] = AngularInterpolation(mat_int, imtype, arg3, pinterpflag);
%
% Last modified by KA dd061026
% recon_gui version 0.1

nplr=size(mat_int,4); naz=size(mat_int,3); nspec=size(mat_int,2);

% for compatibility to old code (before Aug 2006)
if length(arg3) > 1, % new protocol
    ifactor=0.5*(arg3(1)/nplr+arg3(2)/nspec);
    nAng_i=arg3;
else    % old protocol
    ifactor=arg3;
    nAng_i=ifactor*[nplr nspec];
end

if ifactor ~= 1
    if pinterpflag==1
        disp('Interpolate additional projections - sinc');
        [mat_int,nplr,naz,nspec]=interp_xdsbp(mat_int,imtype,ifactor);
    elseif pinterpflag > 1
        [mat_int,nplr,naz,nspec]=local_interp(mat_int,nAng_i,pinterpflag);
        fprintf('Interpolated to %d polar x %d azimuthal x %d spectral\n',nplr,naz,nspec);
    else
	return;
    end
end

%--------------------------------------------------------------------------
function [mat_int,nplr,naz,nspec]=interp_xdsbp(mat_int,imtype,ifactor);
%  INTERPOLATE DATA TO INCREASE THE NUMBER OF PROJECTIONS
%	The upshot is, the angular separations are reduced, but the initial angle remains the same.
nplr=size(mat_int,4);
naz=size(mat_int,3);
nspec=size(mat_int,2);
nbins=size(mat_int,1);

if nspec>1, nspec=nspec*ifactor;end
if naz>1, naz=naz*ifactor;end
if nplr>1, nplr=nplr*ifactor;end

if naz==1
    nAng1=nspec; nAng2=nplr;
elseif nplr==1
    nAng1=nspec; nAng2=naz;
elseif nspec==1
    nAng1=naz; nAng2=nplr;
end

switch imtype
    case 1 % 4D
        for i=1:nbins
            mat_int_i(i,:,:,:)=reshape(zp3D(squeeze(mat_int(i,:,:,:)),ifactor),1,nspec,naz,nplr);
        end
    case {2,3,4,14} % 3D
        for i=1:nbins
            mat_int_i(i,:,:)=reshape(zp2D(squeeze(mat_int(i,:,:,:)),ifactor),1,nAng1,nAng2);
        end
    case {5,6,7,11,12,13} % 2D
        mat_int_i=interp1Dsinc(squeeze(mat_int),ifactor);
    otherwise
        disp('Angular interpolation ignored for 1D image');
        return;
end
mat_int=reshape(mat_int_i,nbins,nspec,naz,nplr); clear mat_int_i;
% END OF function [mat_int,nplr,naz,nspec]=interp_xdsbp(mat_int,imtype,ifactor);

%--------------------------------------------------------------------------
function [mat_int, nPolar_i, nAz_i, nSpec_i]=local_interp(mat_int,nAng_i,pinterpflag);

switch pinterpflag
    case {2,4,6,2.5} %-------------
        disp('spline angular interpolation');
        methodStr='spline';
    case {3,5}
        disp('linear angular interpolation');
        methodStr='linear';
end

% For spectral angles outside sampled region,
% if pinterpflag 2, 3, interpolate using symmetry
% if pinterpflag 4, 5, extrapolate
if pinterpflag > 3
    extrapOn=1;
    if pinterpflag==4
        disp('spline spectral angular extrapolation');
    else
        disp('linear spectral angular extrapolation');
    end
else
    extrapOn=0;
         disp('spectral angular extrapolation is replaced with data mirroring');
end
    
% mat_int(nbins,nSpec,nAz,nPolar)
nbins=size(mat_int,1);
nSpec=size(mat_int,2);
nAz=size(mat_int,3);
nPolar=size(mat_int,4);
dataTypeStr=class(mat_int);

[alpha0, alpha, alpha_i, phi, phi_i, theta, theta_i, nSpec_i, nAz_i, nPolar_i]=...
        CalcInterpolationAngles(nSpec,nAz,nPolar,nAng_i);

mat_int_i1=zeros(nbins,nSpec_i,nAz,nPolar,dataTypeStr);
mat_int_i2=zeros(nbins,nSpec_i,nAz_i,nPolar_i,dataTypeStr);
%------------
mat_add2=zeros(nbins,nSpec_i,nAz_i,nPolar+2,dataTypeStr);
%------------

x=1:nbins;
mat_add=AddSpectralEdge(mat_int);
if extrapOn
    [t1 t2]=meshgrid(alpha0,x);
else
    [t1 t2]=meshgrid(alpha,x);    
end
    [w1 w2]=meshgrid(alpha_i,x);

if nSpec > 1
    for i=1:nAz, for j=1:nPolar
        if extrapOn
            mat_t=squeeze(mat_int(:,:,i,j));
            if pinterpflag==4
                mat_int_i1(:,:,i,j)=interp2(t1,t2,mat_t,w1,w2,methodStr);
            else
                for k=1:size(w1,1) % nbins
                    mat_int_i1(k,:,i,j)=interp1(alpha0,mat_t(k,:),alpha_i,'linear','extrap');
                end
            end
        else
            mat_t=squeeze(mat_add(:,:,i,j));
            mat_int_i1(:,:,i,j)=interp2(t1,t2,mat_t,w1,w2,methodStr);
        end
    end, end
else
    mat_int_i1=mat_int;
end

mat_add=AddSpatialEdge(mat_int_i1);
[t1 t2]=meshgrid(theta,phi);
[w1 w2]=meshgrid(theta_i,phi_i);
for i=1:nbins, for j=1:nSpec_i,
    mat_t=squeeze(mat_add(i,j,:,:));
    if nPolar > 1 & nAz > 1
        if pinterpflag==2.5
            for k=1:nPolar+2
                mat_t=squeeze(mat_add(i,j,:,k));
                mat_add2(i,j,:,k)=interp1(phi,mat_t,phi_i,methodStr);
            end
            for k=1:nAz_i
                mat_t=squeeze(mat_add2(i,j,k,:));
                mat_int_i2(i,j,k,:)=interp1(theta,mat_t,theta_i,methodStr);
            end
        else
            mat_int_i2(i,j,:,:)=interp2(t1,t2,mat_t,w1,w2,methodStr);
        end
    elseif nPolar > 1
        mat_t=squeeze(mat_add(i,j,2,:));
        mat_int_i2(i,j,1,:)=interp1(theta,squeeze(mat_t),theta_i,methodStr);
    elseif nAz > 1
        mat_t=squeeze(mat_add(i,j,:,2));
        mat_int_i2(i,j,:,1)=interp1(phi,squeeze(mat_t),phi_i,methodStr);
    else
        mat_int_i2=mat_int_i1;
    end
end, end
mat_int=mat_int_i2;
% END OF function local_interp

function mat_add=AddSpectralEdge(mat)
% For the interpolation at the edge of alpha,
% alpha(1), alpha(2), ..., alpha(nSpec),
% alpha(0) and alpha(nSpec+1) need to be provided.
% From symmetry,
% alpha(0)=alpha(1), and alpha(nSpec+1)=alpha(nSpec);
dataTypeStr=class(mat);
nbins=size(mat,1);
nSpec=size(mat,2);nAz=size(mat,3);nPolar=size(mat,4);
mat_add=zeros(nbins,nSpec+2,nAz,nPolar,dataTypeStr);
mat_add(:,2:nSpec+1,:,:)=mat;
mat_add(:,1,:,:)=mat(:,1,:,:);
mat_add(:,nSpec+2,:,:)=mat(:,nSpec,:,:);
% END OF AddSpectralEdge

function mat_add=AddSpatialEdge(mat)
% For the interpolation at the edge of phi,
% [alpha(m),phi(1),theta(n)],[alpha(m),phi(2),theta(n)],...[alpha(m),phi(nAz),theta(n)]
%   where m=1,2,...nSpec, and n=1,2,...nPolar,
% [alpha(m),phi(0),theta(n)] and [alpha(m),phi(nAz+1),theta(n)] need to be provided.
% From symmetry,
%   [alpha(m),phi(0),theta(n)]=[alpha(m),phi(nAz),theta(nPolar-n+1)]
%   [alpha(m),phi(nAz+1),theta(k)]=[alpha(m),phi(1),theta(nPolar-n+1)]
% 
% For the interpolation at the edge of theta,
% [alpha(m),phi(n),theta(1)],[alpha(m),phi(n),theta(2)],...[alpha(m),phi(n),theta(nPolar)]
%   where m=1,2,...nSpec, and n=1,2,...nAz,
% [alpha(m),phi(n),theta(0)] and [alpha(m),phi(n),theta(nPolar+1)] need to be provided.
% From symmetry,
%   [alpha(m),phi(n),theta(0)]=[alpha(nSpec-m+1),phi(n),theta(nPolar)]
%   [alpha(m),phi(n),theta(nPolar+1)]=[alpha(nSpec-m+1),phi(n),theta(1)]
%
% For the interpolation at the corner,
%   [alpha(m),phi(0),theta(0)]=[alpha(nSpec-m+1),phi(nAz),theta(0)]
%   [alpha(m),phi(0),theta(nPolar+1)]=[alpha(nSpec-m+1),phi(nAz),theta(nPolar)]
%   [alpha(m),phi(nAz+1),theta(0)]=[alpha(nSpec-m+1),phi(0),theta(0)]
%   [alpha(m),phi(nAz+1),theta(nPolar+1)]=[alpha(nSpec-m+1),phi(0),theta(nPolar)]

dataTypeStr=class(mat);
nbins=size(mat,1);
nSpec=size(mat,2);nAz=size(mat,3);nPolar=size(mat,4);
N=nAz+2; M=nPolar+2;
mat_add=zeros(nbins,nSpec,N,M,dataTypeStr);
mat_add(:,:,2:N-1,2:M-1)=mat;
mat_add(:,:,1,2:M-1)=flipdim(mat(:,:,nAz,:),4);% edge of phi(0)
mat_add(:,:,N,2:M-1)=flipdim(mat(:,:,1,:),4);  % edge of phi(nAz+1)
mat_add(:,:,2:N-1,1)=flipSpec(mat(:,:,:,nPolar)); % edge of theta(0)
mat_add(:,:,2:N-1,M)=flipSpec(mat(:,:,:,1));  % edge of theta(nPolar+1)
mat_add(:,:,1,1)=flipSpec(mat(:,:,nAz,1)); % corner of phi(0),theta(0)
mat_add(:,:,1,M)=flipSpec(mat(:,:,nAz,nPolar)); % corner of phi(0),theta(nPolar+1)
mat_add(:,:,N,1)=flipSpec(mat(:,:,1,1)); % corner of phi(nAz+1),theta(0)
mat_add(:,:,N,M)=flipSpec(mat(:,:,1,nPolar)); % corner of phi(nAz+1),theta(nPolar+1)
% END OF AddSpatialEdge

function matOut=flipSpec(mat)
nSpec=size(mat,2);
if nSpec==1
    matOut=flipdim(mat,1);
else
    matOut=flipdim(mat,2);
end
% END OF flipSpec

function [alpha0, alpha, alpha_i, phi, phi_i, theta, theta_i, nSpec_i, nAz_i, nPolar_i]=...
        CalcInterpolationAngles(nSpec,nAz,nPolar,nAng_i)
a=pi/2-pi/2/nSpec; d=pi/nSpec;
alpha0 = a-d*(0:nSpec-1);
alpha = a-d*(-1:nSpec);
a=pi/2-pi/2/nAz; d=pi/nAz;
phi = a-d*(-1:nAz);
a=pi/2-pi/2/nPolar; d=pi/nPolar;
theta = a-d*(-1:nPolar);
nPolar_i=nAng_i(1); nAz_i=nAng_i(1); nSpec_i=nAng_i(2);
if nSpec > 1
    a=pi/2-pi/2/nSpec_i; d=pi/nSpec_i;
    alpha_i=a-d*(0:nSpec_i-1);
else
    alpha_i=alpha;
    nSpec_i=1;
end
if nAz > 1
    a=pi/2-pi/2/nAz_i; d=pi/nAz_i;
    phi_i = a-d*(0:nAz_i-1);
else
    phi_i=phi;
    nAz_i=1;
end
if nPolar > 1
    a=pi/2-pi/2/nPolar_i; d=pi/nPolar_i;
    theta_i=a-d*(0:nPolar_i-1);
else
    theta_i=theta;
    nPolar_i=1;
end
% END OF function CalcInterpolationAngles

%--------------------------------------------------------------------------
function Linput=zp2D(input,pdng);
% interpolation sub routine
nbins1=size(input,1);
nbins2=size(input,2);
dataTypeStr=class(input);
%Make the image periodic
Pinput=zeros(nbins1*2,nbins2*2,dataTypeStr);
Pinput(1:nbins1,1:nbins2)=input;
Pinput(1:nbins1,nbins2+1:2*nbins2)=fliplr(input);
Pinput(nbins1+1:2*nbins1,1:nbins2)=flipud(input);
Pinput(nbins1+1:2*nbins1,nbins2+1:2*nbins2)=fliplr(flipud(input));

%Do the FFTs and zero pad
Bfft=zeros(nbins1*2*pdng,nbins2*2*pdng,dataTypeStr);
Lfft=fft2(Pinput);
%Upper Left
Bfft(1:nbins1,1:nbins2)=Lfft(1:nbins1,1:nbins2);
Bfft(nbins1+1,1:nbins2+1)=.5*Lfft(nbins1+1,1:nbins2+1);
Bfft(1:nbins1+1,nbins2+1)=.5*Lfft(1:nbins1+1,nbins2+1);
%Upper Right
Bfft(1:nbins1,nbins2*2*pdng-nbins2+2:nbins2*2*pdng)=Lfft(1:nbins1,nbins2+2:nbins2*2);
Bfft(nbins1+1,nbins2*2*pdng-nbins2+1:nbins2*2*pdng)=.5*Lfft(nbins1+1,nbins2+1:nbins2*2);
Bfft(1:nbins1+1,nbins2*2*pdng-nbins2+1)=.5*Lfft(1:nbins1+1,nbins2+1);
%Lower Left
Bfft(nbins1*2*pdng-nbins1+2:nbins1*2*pdng,1:nbins2)=Lfft(nbins1+2:nbins1*2,1:nbins2);
Bfft(nbins1*2*pdng-nbins1+1,1:nbins2+1)=.5*Lfft(nbins1+1,1:nbins2+1);
Bfft(nbins1*2*pdng-nbins1+1:nbins1*2*pdng,nbins2+1)=.5*Lfft(nbins1+1:nbins1*2,nbins2+1);
%Lower Right
Bfft(nbins1*2*pdng-nbins1+2:nbins1*2*pdng,nbins2*2*pdng-nbins2+2:nbins2*2*pdng)=Lfft(nbins1+2:nbins1*2,nbins2+2:nbins2*2);
Bfft(nbins1*2*pdng-nbins1+1,nbins2*2*pdng-nbins2+1:nbins2*2*pdng)=.5*Lfft(nbins1+1,nbins2+1:nbins2*2);
Bfft(nbins1*2*pdng-nbins1+1:nbins1*2*pdng,nbins2*2*pdng-nbins2+1)=.5*Lfft(nbins1+1:nbins1*2,nbins2+1);

Binput=zeros(nbins1*2*pdng,nbins2*2*pdng,dataTypeStr);
Binput=real(ifft2(Bfft));
Linput=zeros(nbins1*pdng,nbins2*pdng,dataTypeStr);
%Linput=Binput(nbins*pdng/2+1:2*pdng*nbins-nbins*pdng/2,nbins*pdng/2+1:2*pdng*nbins-nbins*pdng/2);
Linput=Binput(1:nbins1*pdng,1:nbins2*pdng);
Linput=pdng^2*Linput;
% END OF function Linput=zp2D(input,pdng);

%--------------------------------------------------------------------------
function Loutput=zp3D(input,pdng);
% Loutput=zp3D(input,pdng);
% input is the 3D array whose size we wish to increase though interpolating new points
% pdng is the scale by which the size will be increased along all 3 dimensions
% Loutput is a 3D array containing the original points and the new, interpolated points
%   input(x,y,z)=Loutput(x*pdng-1,y*pdng-1,z*pdng-1);
%   all other points in Loutput are interpolated
% Care must be taken when using Loutput in further processing, e.g. image reconstruction,
%   to account for the proper position of the data points.
% This routine was originally written to interpolate new projections in a 4D-sinogram, without
%   increasing the number of bins, or samples per projection.
% Here, let P(bin,theta,phi,alpha) be the full set of acquired data
%   If squeeze(P(N,1:ntheta,1:nphi,1:nalpha)) is the input, the Loutput will be 
%   P(N,1:pdng*ntheta,1:pdng*nphi,1:pdng*nalpha).
% Repeating this for N=1:nbins, will increase the number of projections by pdng^3 

% gets the dimensions of the input array
nbins=size(input);
dataTypeStr=class(input);
% Make the image periodic
% Put he original array in each corner of an array
% that is 2*nbins(1)-by-2*nbins(2)
% use flips so the (nbins(1),nbins(2)) pixel is always next to the center
Pinput=zeros(nbins(1)*2,nbins(2)*2,nbins(3)*2,dataTypeStr);
Pinput(1:nbins(1),1:nbins(2),1:nbins(3))=input;
Pinput(1:nbins(1),nbins(2)+1:2*nbins(2),1:nbins(3))=flipdim(input,2);
Pinput(nbins(1)+1:2*nbins(1),1:nbins(2),1:nbins(3))=flipdim(input,1);
Pinput(nbins(1)+1:2*nbins(1),nbins(2)+1:2*nbins(2),1:nbins(3))=flipdim(flipdim(input,1),2);
Pinput(1:nbins(1),1:nbins(2),nbins(3)+1:2*nbins(3))=flipdim(input,3);
Pinput(1:nbins(1),nbins(2)+1:2*nbins(2),nbins(3)+1:2*nbins(3))=flipdim(flipdim(input,2),3);
Pinput(nbins(1)+1:2*nbins(1),1:nbins(2),nbins(3)+1:2*nbins(3))=flipdim(flipdim(input,1),3);
Pinput(nbins(1)+1:2*nbins(1),nbins(2)+1:2*nbins(2),nbins(3)+1:2*nbins(3))=flipdim(flipdim(flipdim(input,1),2),3);

% Bfft is a new array for the padded FFT
Bfft=zeros(nbins(1)*2*pdng,nbins(2)*2*pdng,nbins(3)*2*pdng,dataTypeStr);
% Lfft is the fft of the, now, periodic input array
Lfft=fftn(Pinput);

% Fill the corners of Bfft appropriately with the corners of Lfft, with the '.5' planes also. 
% (0,0,0)
Bfft(1:nbins(1),1:nbins(2),1:nbins(3))=Lfft(1:nbins(1),1:nbins(2),1:nbins(3));
Bfft(nbins(1)+1,1:nbins(2)+1,1:nbins(3)+1)=.5*Lfft(nbins(1)+1,1:nbins(2)+1,1:nbins(3)+1);
Bfft(1:nbins(1)+1,nbins(2)+1,1:nbins(3)+1)=.5*Lfft(1:nbins(1)+1,nbins(2)+1,1:nbins(3)+1);
Bfft(1:nbins(1)+1,1:nbins(2)+1,nbins(3)+1)=.5*Lfft(1:nbins(1)+1,1:nbins(2)+1,nbins(3)+1);

% (0,1,0)
Bfft(1:nbins(1),nbins(2)*2*pdng-nbins(2)+2:nbins(2)*2*pdng,1:nbins(3))=Lfft(1:nbins(1),nbins(2)+2:nbins(2)*2,1:nbins(3));
Bfft(nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,1:nbins(3)+1)=.5*Lfft(nbins(1)+1,nbins(2)+1:nbins(2)*2,1:nbins(3)+1);
Bfft(1:nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1,1:nbins(3)+1)=.5*Lfft(1:nbins(1)+1,nbins(2)+1,1:nbins(3)+1);
Bfft(1:nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,nbins(3)+1)=.5*Lfft(1:nbins(1)+1,nbins(2)+1:nbins(2)*2,nbins(3)+1);

% (1,0,0)
Bfft(nbins(1)*2*pdng-nbins(1)+2:nbins(1)*2*pdng,1:nbins(2),1:nbins(3))=Lfft(nbins(1)+2:nbins(1)*2,1:nbins(2),1:nbins(3));
Bfft(nbins(1)*2*pdng-nbins(1)+1,1:nbins(2)+1,1:nbins(3)+1)=.5*Lfft(nbins(1)+1,1:nbins(2)+1,1:nbins(3)+1);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,nbins(2)+1,1:nbins(3)+1)=.5*Lfft(nbins(1)+1:nbins(1)*2,nbins(2)+1,1:nbins(3)+1);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,1:nbins(2)+1,nbins(3)+1)=.5*Lfft(nbins(1)+1:nbins(1)*2,1:nbins(2)+1,nbins(3)+1);

% (1,1,0)
Bfft(nbins(1)*2*pdng-nbins(1)+2:nbins(1)*2*pdng,nbins(2)*2*pdng-nbins(2)+2:nbins(2)*2*pdng,1:nbins(3))=Lfft(nbins(1)+2:nbins(1)*2,nbins(2)+2:nbins(2)*2,1:nbins(3));
Bfft(nbins(1)*2*pdng-nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,1:nbins(3)+1)=.5*Lfft(nbins(1)+1,nbins(2)+1:nbins(2)*2,1:nbins(3)+1);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,nbins(2)*2*pdng-nbins(2)+1,1:nbins(3)+1)=.5*Lfft(nbins(1)+1:nbins(1)*2,nbins(2)+1,1:nbins(3)+1);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,nbins(3)+1)=.5*Lfft(nbins(1)+1:nbins(1)*2,nbins(2)+1:nbins(2)*2,nbins(3)+1);

% (0,0,1)
Bfft(1:nbins(1),1:nbins(2),nbins(3)*2*pdng-nbins(3)+2:nbins(3)*2*pdng)=Lfft(1:nbins(1),1:nbins(2),nbins(3)+2:nbins(3)*2);
Bfft(nbins(1)+1,1:nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(nbins(1)+1,1:nbins(2)+1,nbins(3)+1:nbins(3)*2);
Bfft(1:nbins(1)+1,nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(1:nbins(1)+1,nbins(2)+1,nbins(3)+1:nbins(3)*2);
Bfft(1:nbins(1)+1,1:nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1)=.5*Lfft(1:nbins(1)+1,1:nbins(2)+1,nbins(3)+1);

% (0,1,1)
Bfft(1:nbins(1),nbins(2)*2*pdng-nbins(2)+2:nbins(2)*2*pdng,nbins(3)*2*pdng-nbins(3)+2:nbins(3)*2*pdng)=Lfft(1:nbins(1),nbins(2)+2:nbins(2)*2,nbins(3)+2:nbins(3)*2);
Bfft(nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(nbins(1)+1,nbins(2)+1:nbins(2)*2,nbins(3)+1:nbins(3)*2);
Bfft(1:nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(1:nbins(1)+1,nbins(2)+1,nbins(3)+1:nbins(3)*2);
Bfft(1:nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,nbins(3)*2*pdng-nbins(3)+1)=.5*Lfft(1:nbins(1)+1,nbins(2)+1:nbins(2)*2,nbins(3)+1);

% (1,0,1)
Bfft(nbins(1)*2*pdng-nbins(1)+2:nbins(1)*2*pdng,1:nbins(2),nbins(3)*2*pdng-nbins(3)+2:nbins(3)*2*pdng)=Lfft(nbins(1)+2:nbins(1)*2,1:nbins(2),nbins(3)+2:nbins(3)*2);
Bfft(nbins(1)*2*pdng-nbins(1)+1,1:nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(nbins(1)+1,1:nbins(2)+1,nbins(3)+1:nbins(3)*2);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(nbins(1)+1:nbins(1)*2,nbins(2)+1,nbins(3)+1:nbins(3)*2);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,1:nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1)=.5*Lfft(nbins(1)+1:nbins(1)*2,1:nbins(2)+1,nbins(3)+1);

% (1,1,1)
Bfft(nbins(1)*2*pdng-nbins(1)+2:nbins(1)*2*pdng,nbins(2)*2*pdng-nbins(2)+2:nbins(2)*2*pdng,nbins(3)*2*pdng-nbins(3)+2:nbins(3)*2*pdng)=Lfft(nbins(1)+2:nbins(1)*2,nbins(2)+2:nbins(2)*2,nbins(3)+2:nbins(3)*2);
Bfft(nbins(1)*2*pdng-nbins(1)+1,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(nbins(1)+1,nbins(2)+1:nbins(2)*2,nbins(3)+1:nbins(3)*2);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,nbins(2)*2*pdng-nbins(2)+1,nbins(3)*2*pdng-nbins(3)+1:nbins(3)*2*pdng)=.5*Lfft(nbins(1)+1:nbins(1)*2,nbins(2)+1,nbins(3)+1:nbins(3)*2);
Bfft(nbins(1)*2*pdng-nbins(1)+1:nbins(1)*2*pdng,nbins(2)*2*pdng-nbins(2)+1:nbins(2)*2*pdng,nbins(3)*2*pdng-nbins(3)+1)=.5*Lfft(nbins(1)+1:nbins(1)*2,nbins(2)+1:nbins(2)*2,nbins(3)+1);

% Boutput is the for the ifft of Bfft
Boutput=zeros(nbins(1)*2*pdng,nbins(2)*2*pdng,nbins(3)*2*pdng,dataTypeStr);
% Inverse transform Bfft, store in Boutput
Boutput=real(ifftn(Bfft));
% Loutput is a cropped section of Boutput, it is the output
Loutput=zeros(nbins(1)*pdng,nbins(2)*pdng,nbins(3)*pdng,dataTypeStr);
% Fill Loutput with the (0,0,0) corner of Boutput
Loutput=Boutput(1:nbins(1)*pdng,1:nbins(2)*pdng,1:nbins(3)*pdng);
% Normalize the output
Loutput=pdng^3*Loutput;
% END OF function Loutput=zp3D(input,pdng);