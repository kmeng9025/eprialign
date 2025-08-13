% IRADON_FBPGRADTABLE uniform angular and uniform spatial gradient table
% for FBP images
% [out, suplementary_out] = iradon_FBP_grad_table(FBP)
% FBP - [structure] of gradient scheme parameters
%   [].imtype - Image type [int, 1 for 4D, 14 for 3D]
%   [].nPolar - Number of polar angles [int]
%   [].nAz    - Number of azimuthal angles [int]
%   [].nSpec  - Number of spectral angles[int]
%   [].size - length of the spatial projection [float, in cm]
%   [].CoordPole - [1/2/3] or <X/Y/(Z)>
%   [].MaxGradient - Maximum gradient [float, in G/cm]
%   [].angle_sampling - type of angle sampling
%       UNIFORM_ANGULAR  - uniform angular
%       UNIFORM_ANGULAR_FLIP - uniform angular with optimized jumps
%       UNIFORM_SPATIAL_FLIP - uniform solid angle with optim. jumps
% out - [structure] of radon transformation parameters
%   [].GradX - Gradient component [array, 1D]
%   [].GradY - Gradient component [array, 1D]
%   [].GradZ - Gradient component [array, 1D]
% suplementary_out - [structure] of radon transformation parameters
%   [].kx - k-space unit vector component [array, 1D]
%   [].ky - k-space unit vector component [array, 1D]
%   [].kz - k-space unit vector component [array, 1D]
%   [].w  - projection weight factor [array, 1D]
% 
% See also IRADON_GETFBPIMAGETYPE

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago,JULY 2013
% Contact: epri.uchicago.edu

function [out, suplementary_pars] = iradon_FBPGradTable(FBP)

[imtype, cimtype] = iradon_GetFBPImageType(FBP.imtype);
coordPole = iradon_GetCoordPole(safeget(FBP, 'CoordPole','Z'));
out.nPolar=safeget(FBP,'nPolar',16);
out.nAz=safeget(FBP,'nAz',16);
gradR = safeget(FBP, 'MaxGradient', 1);
out.nSpec=safeget(FBP,'nSpec',14);
angle_sampling = upper(safeget(FBP,'angle_sampling','uniform_spatial'));
out.N = 1;

switch cimtype
  case 'XY',      out.nPolar = 1; aalpha = pi/4; out.nSpec = 1;
  case 'XYB'      
    out.nAz = 1; 
    aalpha = pi/2+pi/out.nSpec/2-pi/out.nSpec*(1:out.nSpec);
  case {'XZB', 'YZB'}
    out.nAz = 1; 
    aalpha = pi/2+pi/out.nSpec/2-pi/out.nSpec*(1:out.nSpec);
  case {'YZ', 'XZ'} ,out.nAz = 1; aalpha = pi/4; out.nSpec = 1;
  case 'XYZ', aalpha = pi/4; out.nSpec = 1;
  case {'XB', 'YB', 'ZB'}
    out.nAz = 1; 
    out.nPolar = 1;
    aalpha = pi/2+pi/out.nSpec/2-pi/out.nSpec*(1:out.nSpec);
  otherwise
    aalpha = pi/2+pi/out.nSpec/2-pi/out.nSpec*(1:out.nSpec);
end

% Dimensions of the experiment
out.Dim = [out.nSpec,out.nAz,out.nPolar];
out.gidx = reshape(1:prod(out.Dim), out.Dim);

% low dimensions imaging
[firstTheta, firstPhi] = getFirstAngle(imtype,out.nPolar,out.nAz);
if out.nAz==1,    phi=firstPhi; end

% spectral angle
all_alpha = repmat(aalpha', [1,out.nAz, out.nPolar]);
suplementary_pars.k  = repmat((1:out.nSpec)', [1,out.nAz, out.nPolar]);

% polar angle
if out.nPolar == 1, theta=firstTheta;
else, theta=pi/2+pi/out.nPolar/2-pi/out.nPolar*(1:out.nPolar);
end
all_theta     = repmat(reshape(theta, [1,1,out.nPolar]), [out.nSpec,out.nAz]);
suplementary_pars.i  = repmat(reshape(1:out.nPolar, [1,1,out.nPolar]), [out.nSpec,out.nAz]);

% azimuthal angle
all_phi       = zeros(out.nSpec,out.nAz,out.nPolar);

% Different schemes
switch angle_sampling
  case 'UNIFORM_ANGULAR'
    phi=(pi/2+pi/out.nAz/2-pi/out.nAz*(1:out.nAz));
    all_phi       = repmat(phi, [out.nSpec,1,out.nPolar]);
    suplementary_pars.j  = repmat(reshape(1:out.nPolar, [1,1,out.nPolar]), [out.nSpec,1,out.nPolar]);
    
    delta_theta = pi/out.nPolar;
    delta_phi   = pi/out.nAz;
    suplementary_pars.w=delta_theta*delta_phi.*sin(all_theta);    
  case 'UNIFORM_ANGULAR_FLIP'
    phi=(pi/2+pi/out.nAz/2-pi/out.nAz*(1:out.nAz));
    all_phi     = repmat(phi, [out.nSpec,1,out.nPolar]);
    suplementary_pars.j  = repmat(reshape(1:out.nAz, [1,1,out.nPolar]), [out.nSpec,1,out.nPolar]);
    
    delta_theta = pi/out.nPolar;
    delta_phi   = pi/out.nAz;
    suplementary_pars.w=delta_theta*delta_phi.*abs(sin(all_theta));

    all_alpha  = alterate_order(all_alpha, 1);
    all_phi    = alterate_order(all_phi, 2);
    suplementary_pars.w  = alterate_order(suplementary_pars.w, 2);
    
    out.gidx    = alterate_order(out.gidx, 1);
    out.gidx    = alterate_order(out.gidx, 2);

    suplementary_pars.k  = alterate_order(suplementary_pars.k, 1);
    suplementary_pars.j = alterate_order(suplementary_pars.j, 2);
    
  case 'UNIFORM_SPATIAL_FLIP'
    if out.nAz == 1
      all_phi(:)= 0; 
      suplementary_pars.j = repmat(1:out.nPolar, [out.nSpec,1]);
    else
      n_polar_angles = round(abs(out.nPolar*sin(theta)));
      if out.nPolar > 1
        for ii=1:out.nPolar
          add = n_polar_angles(ii);
          phi=(pi/2+pi/add/2-pi/add*(1:add));
          all_phi(:,1:add,ii)      = repmat(phi, [out.nSpec,1]);
          suplementary_pars.j(:,1:add,ii) = repmat(1:add, [out.nSpec,1]);
          out.gidx(:,add+1:end,ii)= 0;
        end
        suplementary_pars.j = alterate_order(suplementary_pars.j, 2);
        all_phi     = alterate_order(all_phi, 2);
      else
        phi=(pi/2+pi/out.nAz/2-pi/out.nAz*(1:out.nAz));
        all_phi = repmat(phi', [1, out.nSpec])';
      end
    end
    
    delta_theta=pi/out.nPolar;
    suplementary_pars.w=delta_theta^2*ones(size(all_theta));
    
    out.gidx    = alterate_order(out.gidx, 2);
    suplementary_pars.w       = alterate_order(suplementary_pars.w, 2);
    
    all_alpha    = alterate_order_part(all_alpha, 1, out.gidx);
    suplementary_pars.k  = alterate_order_part(suplementary_pars.k, 1, out.gidx);
    suplementary_pars.w        = alterate_order_part(suplementary_pars.w, 1, out.gidx);
    out.gidx     = alterate_order_part(out.gidx, 1, out.gidx);
  case 'EQUAL_SOLID_ANGLE_GAGE'    
    nphi = FBP.nPhi;
    ntheta = FBP.nTheta; 
    [all_theta,all_phi,out, suplementary_pars] = ...
        mingrad_minstddev(nphi,ntheta,out.nPolar,out.nAz);
    all_alpha = (pi/4)*ones(size(all_theta));
  case 'UNIFORM_SPATIAL_SPIRAL_8Q'
    theta = theta + pi/2;
    all_theta     = reshape(repmat(theta', [1,out.nPolar*2])', [1,out.nPolar*2,out.nAz]);
    out.Dim = [out.nSpec,out.nAz*2,out.nPolar];
    out.gidx = reshape(1:prod(out.Dim), out.Dim);
    all_alpha = aalpha*ones(out.Dim);
    
    suplementary_pars.i  = reshape(repmat((1:out.nPolar*2)', [1,out.nAz])', [1,out.nPolar*2,out.nAz]);
    suplementary_pars.k = ones(out.Dim);


    n_polar_angles = round(abs(out.nPolar*sin(theta)))*2;
    for ii=1:out.nPolar
      add = n_polar_angles(ii);
      phi=(pi/2+2*pi/add/2-2*pi/add*(1:add));
      %         phi= -(2*pi)/add*(0.5:add-0.5);
      
      all_phi(:,1:add,ii)      = repmat(phi, [out.nSpec,1]);
      suplementary_pars.j(:,1:add,ii) = repmat(1:add, [out.nSpec,1]);
      out.gidx(:,add+1:end,ii)= 0;
    end
    
    delta_theta=pi/out.nPolar/2;
    suplementary_pars.w=delta_theta^2*ones(size(all_theta));
    
%     out.gidx    = alterate_order(out.gidx, 2);
%     suplementary_pars.w       = alterate_order(suplementary_pars.w, 2);
%     
%     all_alpha    = alterate_order_part(all_alpha, 1, out.gidx);
%     suplementary_pars.k  = alterate_order_part(suplementary_pars.k, 1, out.gidx);
%     suplementary_pars.w        = alterate_order_part(suplementary_pars.w, 1, out.gidx);
%     out.gidx     = alterate_order_part(out.gidx, 1, out.gidx);
  otherwise, return
end

if out.nPolar > 1
  all_alpha = all_alpha(out.gidx > 0);
end
all_theta = all_theta(out.gidx > 0); all_theta = all_theta(:);
all_phi   = all_phi(out.gidx > 0); all_phi = all_phi(:);
out.Dim = out.Dim(out.Dim > 1);

suplementary_pars.w  = suplementary_pars.w(out.gidx > 0);

suplementary_pars.i=suplementary_pars.i(out.gidx > 0);
if out.nPolar > 1
  suplementary_pars.j=suplementary_pars.j(out.gidx > 0);
end
suplementary_pars.k=suplementary_pars.k(out.gidx > 0);

out.gidx = out.gidx(out.gidx > 0);
out.nP = length(out.gidx);
out.nTrace = out.nP;

dBdL = gradR/tan(max(aalpha));
all_tan_aalpha = tan(all_alpha(:));
switch coordPole
  case 1 % x pole
    suplementary_pars.unx=cos(all_theta);
    suplementary_pars.uny=sin(all_theta).*sin(all_phi);
    suplementary_pars.unz=sin(all_theta).*cos(all_phi);
  case 2 % y pole
    suplementary_pars.unx=sin(all_theta).*cos(all_phi);
    suplementary_pars.uny=cos(all_theta);
    suplementary_pars.unz=sin(all_theta).*sin(all_phi);
  case 3 % z pole
    suplementary_pars.unx=sin(all_theta).*cos(all_phi);
    suplementary_pars.uny=sin(all_theta).*sin(all_phi);
    suplementary_pars.unz=cos(all_theta);
end

suplementary_pars.kx=all_tan_aalpha.*suplementary_pars.unx;
suplementary_pars.ky=all_tan_aalpha.*suplementary_pars.uny;
suplementary_pars.kz=all_tan_aalpha.*suplementary_pars.unz;

out.G = dBdL*[suplementary_pars.kx, suplementary_pars.ky, suplementary_pars.kz];
out.service_idx = ones(out.nP,1);

out.deltaL = safeget(FBP, 'deltaL', 3.0*sqrt(2));
out.deltaH = dBdL*out.deltaL;
out.UnitSweep = 1 ./cos(all_alpha);

out.data.Modality = 'PULSEFBP';
out.data.FBP = FBP;

suplementary_pars.alpha = all_alpha;
suplementary_pars.theta = all_theta;
suplementary_pars.phi   = all_phi;

out.nTrace = length(out.service_idx);

[out] = epri_msps(out, FBP);
[out] = epri_navigator(out, safeget(FBP, 'NAV', []));

switch safeget(FBP, 'scheme', 'single_b')
  case 'single_b'
    %[out.aPolar out.aAz suplementary_pars.wt] = getAngles(out.nPolar,out.nAz,out.gidx);
    out = epri_baseline(out, FBP);
  case 'multi_b'
    out.BLoffset = safeget(FBP, 'BLoffset',-9);
    out = epri_baseline(out, FBP);
    % out = SetZeroGradProtocol(out, FBP);
    out = SetFieldOffsetProtocol(out, FBP); % Most of Multi-B action goes here
    % out = SetSweepWidthFractionProtocol(out, FBP);
    %    out = epri_baseline(out, FBP);
    % out = SetZeroGradProtocol(out, FBP);
    %  out = td_SetFieldOffsetProtocol(out, FBP);
    % out = SetSweepWidthFractionProtocol(out, FBP);
  case 'rapid_scan'
    out = epri_baseline(out, FBP);
    out = epri_split_field(out, FBP);
    % add epri_zero_gradient
end



function [firstTheta, firstPhi] = getFirstAngle(imtype,nPolar,nAz)
firstTheta=pi/2-0.5*pi/nPolar;
firstPhi=pi/2-0.5*pi/nAz;
switch imtype
  case 2, firstPhi=0;
  case 3, firstPhi=pi/2;
  case 4, firstTheta=pi/2;
  case 5, firstTheta=pi/2; firstPhi=0;
  case 6, firstTheta=pi/2; firstPhi=pi/2;
  case 7, firstTheta=0; firstPhi=0;
  case 8, firstTheta=pi/2; firstPhi=0;
  case 9, firstTheta=pi/2; firstPhi=pi/2;
  case 10, firstTheta=0; firstPhi=0;
  case 11, firstPhi=0;
  case 12, firstPhi=pi/2;
  case 13, firstTheta=pi/2;
end
% END OF function getFirstAngle

function data = alterate_order(data, dim)

sz = size(data);
if ndims(data) == 3
  switch dim
    case 1
      data = reshape(data, [sz(1),prod(sz(2:end))]);
      data(:,2:2:end) = flipud(data(:, 2:2:end));
      data = reshape(data, sz);
    case 2
      data = permute(data, [2,3,1]);
      data = reshape(data, [sz(2),sz(1)*sz(3)]);
      data(:,2:2:end) = flipud(data(:, 2:2:end));
      data = reshape(data, sz([2,3,1]));
      data = permute(data, [3,1,2]);
  end
else
  switch dim
    case 1
      data = reshape(data, [sz(1),prod(sz(2:end))]);
      data(:,2:2:end) = flipud(data(:, 2:2:end));
      data = reshape(data, sz);
    case 2
      data = permute(data, [2,1]);
      data = reshape(data, [sz(2),sz(1)]);
      data(:,2:2:end) = flipud(data(:, 2:2:end));
      data = reshape(data, sz([2,1]));
      data = permute(data, [2,1]);
  end
end

% general version of alterate_order able to skip the absent indexes
function data = alterate_order_part(data, dim, idx)

sz = size(data);
if ndims(data) == 3
  switch dim
    case 1
      data = reshape(data, [sz(1),prod(sz(2:end))]);
      idx  = reshape(idx, [sz(1),prod(sz(2:end))]);
      idx = find(squeeze(idx(1,:)));
      data(:,idx(2:2:end)) = flipud(data(:, idx(2:2:end)));
      data = reshape(data, sz);
  end
else
  switch dim
    case 1
      idx  = reshape(idx, [sz(1),prod(sz(2:end))]);
      idx = find(squeeze(idx(1,:)));
      data(:,idx(2:2:end)) = flipud(data(:, idx(2:2:end)));
  end  
end