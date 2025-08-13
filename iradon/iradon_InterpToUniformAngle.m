function mat_i=iradon_InterpToUniformAngle(mat,dataType)
% mat_i=InterpToUniformAngle(mat,dataType);
% dataType == 'timeStamp' : do spline interp & extrapolate
% dataType == 'imgData' : do interpolation using azimuthal symmetry
%
% Last modified by KA dd061208
% recon_gui version 0.1

% mat(nbins,nSpec,nAz,nPolar)
nbins=size(mat,1);
nSpec=size(mat,2);
nAz=size(mat,3);
nPolar=size(mat,4);

xi=1:nbins;
d=pi/nPolar; a=pi/2+d/2;
theta = a-d*(1:nPolar);
d=pi/nAz; a=pi/2+d/2;
phi0 = a-d*(1:nAz);

mat_i=zeros(nbins,nSpec,nAz,nPolar);
mat_add=zeros(nbins,nSpec,nAz+2,nPolar);
mat_add(:,:,2:nAz+1,:)=mat;
for k=1:nPolar
  nAz_=max(1,round(abs(sin(theta(k)))*nAz));
  mat_add(:,:,nAz_+2,k)=mat(:,:,1,nPolar-k+1);
  mat_add(:,:,1,k)=mat(:,:,nAz_,nPolar-k+1);
end

%[aa ab]=myMaxMin(mat);
%figure;for j=1:10, subplot(2,5,j);imagesc(squeeze(mat(:,1,:,j)),[ab aa]);end
for j=1:nSpec
  for k=1:nPolar
  	nAz_=max(1,round(abs(sin(theta(k)))*nAz));
  	d=pi/nAz_; a=pi/2+d/2;
  	if contains(dataType,'timeStamp')
      phi=a-d*(1:nAz_);
      for m=1:9  % size(mat_info,1)==9
        mat_i(m,j,:,k)=interp1(phi,squeeze(mat(m,j,1:nAz_,k)),phi0,'spline','extrap');
      end
  	elseif contains(dataType,'imgData')
      phi=a-d*(0:nAz_+1);
      [t1,t2] = meshgrid(phi,xi);
      [w1,w2] = meshgrid(phi0,xi);
      mat_t=squeeze(mat_add(:,j,1:nAz_+2,k));
      mat_i(:,j,:,k)=interp2(t1,t2,mat_t,w1,w2,'spline');
    else
      fprintf('\nError: in InterpToUniformAngle, wrong option for dataType\n\n');
  	end
  end
end
%figure;for j=1:10, subplot(2,5,j);imagesc(squeeze(mat_add(:,1,:,j)),[ab aa]);end
