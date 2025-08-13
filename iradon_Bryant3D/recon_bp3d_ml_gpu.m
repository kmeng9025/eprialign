function [ m3D ] = recon_bp3d_ml_gpu(p,Pars,Rec)

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
 
% GPU variables
  m3Dg = gpuArray(single(m3D));
  gPhi = gpuArray(Rec.Phi);
  gTheta = gpuArray(Rec.Theta);
  gWt = gpuArray(Rec.Wt);
  gX = gpuArray(x);
  gY = gpuArray(y);
  gZ = gpuArray(z);
  gP = gpuArray(p);
  
% Backprojection
  f = waitbar(0,'Backprojecting');
  for ii = 1:Rec.nProj
    [m3Dg] = backproject3D(gPhi(ii),gTheta(ii),gWt(ii),...
      gX,gY,gZ,gP(:,ii),ctrPt,m3Dg);
    geval(m3Dg);
    f = waitbar(ii/Rec.nProj,f);
  end
  close(f)
  m3D = gather(m3Dg);

