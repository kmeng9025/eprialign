 function [ m3D ] = recon_bp3d(p,Pars,Rec)

%---Input---
% p = array of projections in column vectors
% FBP
% num.rec  = nbins of reconstructioned image
%-----------
%
%---Output--
% mat      = 3d matrix of backprojeted image
%-----------


% Pre-interpolation and Zero Padding
  Rec.intrp = 4 * Pars.nBins;
  p = imresize(p, [Rec.intrp Rec.nProj]);
  
% Zero pad the projections to size 1+2*ceil(N/sqrt(2)  
  imgDiag = sqrt(3)*Rec.intrp+1;
  rz = imgDiag - Rec.intrp;
  p = [zeros(ceil(rz/2),size(p,2)); p; zeros(floor(rz/2),size(p,2))];
  p = [p; zeros(1,Rec.nProj)];
 

% Indices & Variables
  m3D = zeros(Rec.nBins,Rec.nBins,Rec.nBins);
  ctrPt = (size(p,1)+1)/2;
  ctrPtIm = (Rec.nBins+1)/2;
  rVec = ((1:Rec.nBins)-ctrPtIm)*Rec.intrp/Rec.nBins;
  [x,y,z] = ndgrid(rVec);
 
% Backprojection
  f = waitbar(0,'Backprojecting');
  for ii = 1:Rec.nProj
    [m3D] = backproject(Rec.Phi(ii),Rec.Theta(ii),Rec.Wt(ii),...
      x,y,z,p(:,ii),ctrPt,m3D);
    f = waitbar(ii/Rec.nProj,f);
  end
  close(f)
 
  %%
  function [m3D] = backproject(Phi,Theta,Wt,x,y,z,p,ctrPt,m3D)
    tp = x.*cos(Phi).*sin(Theta) + y.*sin(Phi).*sin(Theta) + z.*cos(Theta);
    tp = tp(:) + ctrPt + 1;
    a = floor(tp);
    delta = tp - a;
    m3D(:) = m3D(:) + Wt*(delta(:).*p(a(:)+1) + (1-delta(:)).*p(a(:)));