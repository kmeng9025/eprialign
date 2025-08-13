% CW_SRECONSTRUCT  Reconstruction script for sinusoidal Rapid Scan image
% [mat_recFXD, rec_info, dsc] = RS_SRECONSTRUCT(rs, raw_info, rec_info);
% rs         - Projection data, time along columns [array, 2D]
% raw_info   - [structure] of raw data parameters
%     [].FieldSweep  - Array of field sweeps for every projection [array, in G]
%     [].RSfrequency - Array of RS frequencies for every projection [array, in Hz]
%     [].sampling - Array of dwell times for every projection [array, in s]
% rec_info   - [structure] of image parameters
%     [].ppr - [structure] of data processing parameters
%         see RS_SDECONVOLVE/par_struct for more details.
%     [].rec - [structure] of image reconstruction parameters
%         see IRADON_D2D_MSTAGE/recon_pars for more details.
% mat_recFXD - Reconstructed image [array, 3D or 4D]
% See also RS_SFBP, IRADON_D2D_MSTAGE, RS_SDECONVOLVE, RS_SSCAN_PHASE.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013
% Contact: epri.uchicago.edu

function [mat_recFXD, field_pars, dsc] = cw_reconstruct(in_y, raw_info, field_pars, varargin)

if nargin < 3
  error('Usage: [out_y, out_pars] = td_reconstruct(in_y, in_opt, raw_info, rec_info, vargin)');
elseif nargin > 4
  if ~mod(nargin-1,2)
    for kk=1:2:nargin-1
      in_opt.(lower(varargin{kk})) = varargin{kk+1};
    end
  else error('rs_sreconstruct: Wrong amount of arguments')
  end
end

ZEROGRADIENT_INDEX = -1;

dsc = [];
fbp_struct = raw_info.data.FBP;
ppr_struct = field_pars.ppr;
prc_struct = field_pars.prc;

[pars, pars_ext] = iradon_FBPGradTable(fbp_struct);

is_projection_loaded = false;
if isstruct(in_y), is_projection_loaded = true; end

% complain
if ~is_projection_loaded && pars.nTrace ~= size(in_y, 2), error('Wrong number of projections'); end

nBins = field_pars.rec.nBins;
rec_enabled = isequal(safeget(field_pars.rec, 'Enabled','on'), 'on');
GRADIENT_INDEX = -1;

% Spatial image reconstruction
tic
if is_projection_loaded
  
  % determine image dimensions
  tan_alpha = max(tan(pars_ext.alpha));
  deltaB =  pars.data.FBP.MaxGradient * field_pars.rec.size / tan_alpha;
  pars.UnitSweep = pars.UnitSweep(pars.service_idx ~= GRADIENT_INDEX);
  ReconSweep = pars.UnitSweep * deltaB;
  range_center = ppr_struct.data_offset;
  cos_alpha = cos(pars_ext.alpha);

  x_ss = zeros(field_pars.rec.nBins, pars.nP);
  rec_y = zeros(field_pars.rec.nBins, pars.nP);
  ProjectionLayoutIndex = pars.gidx;
  IdxShow  = cell(pars.nSpec, 1);

  % Remove zero-gradients
  B1 = in_y.B(pars.service_idx ~= GRADIENT_INDEX);
  P1 = in_y.P(pars.service_idx ~= GRADIENT_INDEX);
  G1 = in_y.G(pars.service_idx ~= GRADIENT_INDEX, :);
  
  for ii=1:pars.nSpec
    idxSpec = pars_ext.k == ii;
    image_sweep = mean(ReconSweep(idxSpec));
    cos_sweep = mean(cos_alpha(idxSpec));
    
    B = B1(idxSpec);
    P = P1(idxSpec);
    nSpecPrj = length(P);
    
    % For a now ....
    x_ss_scan = B{1};
    out_y_scan = zeros(length(P{1}),nSpecPrj);
    for kk=1:nSpecPrj, out_y_scan(:,kk) = P{kk}; end
    
    % field axis for projections with correct sweep and number of points
    x_ss_sw = linspace(image_sweep/2, -image_sweep/2, field_pars.rec.nBins)+range_center;
    
    % interpolate trace to get correct number of points and
    % normalize spectral intensity
    rec_y(:, idxSpec) = interp1(x_ss_scan, out_y_scan, x_ss_sw, 'pchip', 0) / cos_sweep;
    x_ss(:,  idxSpec) = repmat(x_ss_sw', 1, nSpecPrj);
    IdxShow{ii} = idxSpec;
    
    rec_y(:, idxSpec) = flip(rec_y(:, idxSpec), 1);
  end
  
  % Image field axis
  field_pars.rec.deltaH = deltaB;
  
  prj_stat(x_ss,rec_y,IdxShow, []);
else
  if pars.nSpec == 1
    
    % Index of projections (no zero gradients)
    ProjectionLayoutIndex = pars.gidx;
    % index that defines zero gradients (==-1) or projections (1...pars.nP)
    zero_g = pars.service_idx == GRADIENT_INDEX;
    
    % separate data and zero gradient projections
    out_data  = in_y(:,~zero_g);
    out_zerog = in_y(:,zero_g);
    
    
    bl = mean(out_data);
    out_y = cumsum(out_data - repmat(bl, size(out_data, 1), 1));
    
    FieldMin = mean(raw_info.FieldMin(~zero_g));
    FieldMax = mean(raw_info.FieldMax(~zero_g));
    nPts = size(out_y, 1);
    
    deconpars = ppr_struct;
    
    % Filter projections
    %   Fcut_off = safeget(rec_info.ppr,'Fcut_off', 0)*1E6;
    %   if Fcut_off > 1e3
    %     fs = raw_info.RSfrequency(1) * Npt;
    %     [B,A] = butter(5,2*Fcut_off/fs);
    %     out_y = filter(B,A,out_y);
    %   end
    
    % Interpolate zero field through all projections
    if ~isempty(out_zerog)
      reference_idx  = find(zero_g);
      zero_field_prj = interp1(reference_idx,c_field,1:pars.nTrace);
      zero_field_prj = field_pars.ppr.data_offset;
    else
      zero_field_prj = deconpars.data_offset;
    end
    out_y = real(out_y);
    
    deltaB = fbp_struct.MaxGradient*mean(field_pars.rec.size);
    
    x_prime  = (1:nBins)/(nBins-1) * deltaB; x_prime = x_prime - mean(x_prime);
    for ii=1:pars.nP
      x_ss = linspace(FieldMin, FieldMax, nPts);
      rec_y(:,ii) = interp1(x_ss - zero_field_prj, out_y(:,ii), x_prime, 'linear', 0);
    end
    
    %   needs_flip = rs_get_sweep_dir(zero_gradients);
    %   if any(needs_flip)
%     rec_y = flipud(rec_y);
    %   end
    prj_stat(repmat(x_prime', 1,pars.nP),rec_y,{1:pars.nP}, []);
    
    % reshuffle projections
    nProj = fbp_struct.nPolar;
    nProjM = fix(nProj / 2 + 0.5);
    myidx = zeros(size(rec_y,2), 1);
    for ii=1:nProj
      if ii <= nProjM
        sl_C = nProjM - (ii - 1);
      else
        sl_C = nProj + nProjM - (ii - 1);
      end
      myidx((1:nProj) + (sl_C-1)*nProj) = (1:nProj) + (ii-1)*nProj;
    end
  
    rrrr = zeros(size(rec_y));
    for ii=1:size(rec_y, 2)
      rrrr(:, ii) = rec_y(:, myidx(ii));
    end
    rec_y = rrrr;
  else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%  4D imaging  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    rec_y = zeros(field_pars.rec.nBins, pars.nP);
    x_ss = zeros(field_pars.rec.nBins, pars.nP);
    
    % Index of projections (no zero gradients)
    ProjectionLayoutIndex = pars.gidx;

    % determine image dimensions
    tan_alpha = max(tan(pars_ext.alpha));
    deltaB =  pars.data.FBP.MaxGradient * field_pars.rec.size / tan_alpha;
    ReconSweep = pars.UnitSweep * deltaB;
    
    cos_alpha = cos(pars_ext.alpha);
    ndata = size(in_y, 1);
    for ii=1:pars.nSpec
      % select data for particular spectral angle
      idxSpec = pars_ext.k == ii;
      nSpecPrj = numel(find(idxSpec));
      cos_sweep = mean(cos_alpha(idxSpec));
      sweep = mean(ReconSweep(idxSpec));

      % integrate data
      data4sweep  = cumsum(in_y(:, idxSpec), 1) * sweep/ndata;
      
      x_ss_scan = linspace(-sweep/2, sweep/2, ndata);
      x_ss_sw = linspace(-sweep/2, sweep/2, nBins);
      
      % interpolate trace to get correct number of points and
      % normalize spectral intensity
      rec_y(:, idxSpec) = interp1(x_ss_scan, data4sweep, x_ss_sw, 'pchip', 0) / cos_sweep;
      x_ss(:,  idxSpec) = repmat(x_ss_sw', 1, nSpecPrj);
      IdxShow{ii} = idxSpec;
    end
    
    prj_stat(x_ss,rec_y,IdxShow, []);
    
    % Image field axis
    field_pars.rec.deltaH = deltaB;
   end
end

% copy as is
dsc.raw = in_y;
% MatrixGUI(rec_y);

if ~rec_enabled
  mat_recFXD = [];
  dsc.raw = rec_y;
  return;
end

sz = size(rec_y);
% remove zero gradient traces and use only real part of projections
mat = zeros([sz(1), prod(pars.Dim)]);
rem_idx = ProjectionLayoutIndex >= 0;
mat(:, ProjectionLayoutIndex(rem_idx)) = real(single(rec_y(:,rem_idx)));
mat_out = reshape(mat, [sz(1), pars.Dim]);

% -------------------------------------------------------------------------
% -------------- R E C O N S T R U C T I O N ----------------------------
% -------------------------------------------------------------------------

% reshape array for reconstruction
mat_out = reshape(mat_out, [nBins,fbp_struct.nSpec,fbp_struct.nAz,fbp_struct.nPolar]);

% Interpolate to linear angle
switch safeget(raw_info.data.FBP,'angle_sampling','uniform_spatial')
  case {'uniform_spatial','uniform_spatial_flip'}
    mat_out=iradon_InterpToUniformAngle(mat_out,'imgData');
end

% MatrixGUI(mat_out)

% Reconstruction
radon_pars.ELA =  raw_info.data.FBP;
radon_pars.size = field_pars.rec.size;
recon_pars = field_pars.rec;
mat_recFXD = iradon_d2d_mstage(mat_out, radon_pars, recon_pars);

% normalize amplitude on the unit volume/1D
switch 14
  case 1, n = 4;
  case 14, n = 3; % 3D experiment
end

% Set software scaling factor
field_pars.ampSS = 1 ./ (field_pars.rec.size(1)*0.01/nBins)^(n-1); % meters ;)

if safeget(field_pars.rec, 'DoublePoints', 0)
  switch n
    case 3
      mat_recFXD = reshape(mat_recFXD, [2,nBins,2,nBins,2,nBins]);
      mat_recFXD =  squeeze(sum(sum(sum(mat_recFXD, 1), 3), 5))/8;
  end
end

disp('    Reconstruction is finished.');
field_pars.com = [];

function prj_stat(x,y,idx,opt)
figure(safeget(opt, 'FigFFT', 4)); clf; hold on

nSet = min(length(idx), 7);
pst=epr_CalcAxesPos(nSet, 1, [0.06 0.0005], [0.04 0.05]);

h = zeros(nSet, 1);
for ii = 1:nSet
  h(ii) = axes('Position', pst(ii,:)); hold on
  xset = x(:, idx{ii});
  yset = y(:, idx{ii});
  for jj=1:size(yset, 2)
    ystat = mean(sum(yset));
    sstat_dev = std(sum(yset));
    plot(xset(:,jj), real(yset(:,jj)), 'b');
    text(0.85, 0.8, sprintf('I=%5.3f(%4.3f)', ystat, sstat_dev), 'units', 'normalized')
    %     plot(rx,imag(ry(:,ii,jj)), 'g');
  end
  axis tight
end
set(h(1:end-1), 'XTickLabel', '');
set(h, 'Box', 'on');
xlabel(h(end), '[G]')



