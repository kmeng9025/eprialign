% IRADON_D2D_MSTAGE  reconstuct a 3D or 4D object from 1D projections
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

% Author: Boris Epel
% For authors of reconstruction see inside Recon_AI_Filt_BP.m
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function object=iradon_d2d_mstage(p, radon_pars, recon_pars)

recon_pars.zeropadding=safeget(recon_pars,'zeropadding',1);
M = size(p, 1);

if recon_pars.zeropadding > 1
  [p, recon_pars.zeropadding] = iradon_zeropadding(p, recon_pars.zeropadding);
  com = pars2com(radon_pars, recon_pars);
  
  object = Recon_AI_Filt_BP(p,com);
  
  image_pads=M*(recon_pars.zeropadding-1)/2;
  object=object(image_pads+1:image_pads+M,image_pads+1:image_pads+M,image_pads+1:image_pads+M);
  
  %  adjust the factor to get a right value for the final object
  d_of_proj=radon_pars.size/M;
  object=object/(d_of_proj^2);
else
  com = pars2com(radon_pars, recon_pars);
  object = Recon_AI_Filt_BP(p,com);
  
  object=object/(radon_pars.size/(M-1))^2;
end

% %----------------------------------------------------------------
%  com - Reconstruction parameters
% %----------------------------------------------------------------
% com(1)=imtype 	(image type)
% com(2)=naz    	(number of azimuthal angles)
% com(3)=nbins  	(original number of bins)
% com(4)=nspec  	(number of spectral angles)
% com(5)=naqd   	(number of acquired spectral angles)
% com(6)=halfflag (0: full Acq, 1: half pi)
% com(7)=FHflag	(0: Absorption, 1: first Harmonic)
% com(8)=-1 	(baseline not used)
% com(9)=npolar, naz after Ang. Interpolation
% com(10)=nspec after Ang. Interpolation
% com(11)=dwell time
% com(12)=Lock-in time constant
% com(13)=filtflag (0:ramlak, 1:shepp-logan, 2:cosine, 3:hamming, 4:hann)
% com(14)=not used
% com(15)=0.5 	(cutoff of filter)
% com(16)=order of N(x) for rational fit (N(x)/D(x)) of clearance
% com(17)=order of D(x) for rational fit of clearance
% com(18)=bpCodeFlag (1: Fortran, 2: Matlab, 3: C back projection)
% com(19)=singlePrecision
% com(20)=nplr 	(number of polar angles)
% com(21)=pinterpflag (angular interpolation 0: no interp, 1: sinc, 2: spline, 3: linear)
% com(22)=0,1	(clearance correction)
% com(23)=ssproj	(subsample projections)
% com(24)=ifactor	(angular interpolation factor)
% com(25)=newbins	(new bins after subsampling)
% com(26)=scanMethod
% com(27)=lagshift
% com(28)=swDefCode % 0: sqrt(2)/cos(alpha), 1: 1+tan(alpha)
% %----------------------------------------------------------------
%
% swDefCode=0: SW=sqrt(2)*deltaH/cos(alpha)
% swDefCode=1: SW=deltaH*(1+tan(alpha))
% swDefCode=2: SW=deltaH*(1+k*tan(alpha)), k=ROI/FOV
% swDefCode=3: SW=deltaH*(1+k*tan(alpha)), k varies for grad. direction
%
% scanMethod = a+2b+4c+8d
%  a=1 upscan only, a=0 triangular scan
%  b=1 TC varies,   b=0 N scan varies
%  c=1 no freq. reading, c=0 freq. reading
%  d=1 uniform solid angles, d=0 uniform linear angles
% %----------------------------------------------------------------

function com = pars2com(radon_pars, rec)

% reconstruction parameter
Sub_points = safeget(rec, 'nBins', 64);
if safeget(rec, 'DoublePoints', 0), Sub_points = Sub_points * 2; end

com = zeros(28,1);
com(1) = safeget(radon_pars.ELA, 'imtype', 1);
com(2) = safeget(radon_pars.ELA, 'nAz',  10);
com(3) = Sub_points;
com(4) = safeget(radon_pars.ELA, 'nSpec', 14);
com(5) = safeget(radon_pars.ELA, 'nSpec', 14);
com(6) = 0;
com(7) = 0; % com(7)=FHflag (0: Absorption, 1: first Harmonic)
com(8) = -1;
com(11) = safeget(radon_pars.ELA, 'DwellTime', 0.001);
com(12) = safeget(radon_pars.ELA, 'LockInTimeConst', 0.001);
% Filtering type                                         13
switch(safeget(rec,'Filter','ram-lak'))
  case 'ram-lak', com(13) = 0;
  case 'shepp-logan', com(13) = 1;
  case 'cosine', com(13) = 2;
  case 'hamming', com(13) = 3;
  case 'hann', com(13) = 4;
end
com(14) = 1;
com(15) = safeget(rec,'FilterCutOff',0.5);
% Reconstruction routine                                 18
switch(safeget(rec, 'CodeFlag', 'MATLAB'))
  case 'MATLAB', com(18) = 2;
  case 'FORTRAN', com(18) = 1;
  case 'C', com(18) = 3;
  otherwise, com(18) = 2;
end
com(19) = 1; % single precision
com(20) = safeget(radon_pars.ELA, 'nPolar', 10);
% Interpolation type                                     20
switch(safeget(rec, 'Interpolation', 'sinc'))
  case 'sinc', com(21) = 1;
  case 'spline', com(21) = 2;
  case 'linear', com(21) = 3;
  otherwise, com(21) = 0;
end
% com(22) = raw_info.data.ELA.nSpec; % clearence correction
com(23) = 1;
com(24) = safeget(rec, 'InterpFactor', 1);
com(25) = Sub_points;
com(26) = safeget(radon_pars.ELA, 'scanMethod', 9);
com(27) = 0; % lagshift
com(28) = 0;

