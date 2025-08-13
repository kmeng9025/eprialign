% RS_SRECONSTRUCT  Reconstruction script for sinusoidal Rapid Scan image
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

function [mat_recFXD, field_pars, dsc] = rs_sreconstruct(in_y, raw_info, field_pars, varargin)

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
  % 3D imaging
  if pars.nSpec == 1
    
    % Index of projections (no zero gradients)
    ProjectionLayoutIndex = pars.gidx;
    % index that defines zero gradients (==-1) or projections (1...pars.nP)
    zero_g = pars.service_idx == GRADIENT_INDEX;
    
    % separate data and zero gradient projections
    out_data  = in_y(:,~zero_g);
    out_zerog = in_y(:,zero_g);
    out_freq  = mean(raw_info.RSfrequency(~zero_g));
    out_sweep = mean(raw_info.FieldSweep(~zero_g));
    out_sample = mean(raw_info.sampling(~zero_g));
    
    deconpars = ppr_struct;
    [x_zg,out_zg, deconpars.field_scan_phase]=rs_sscan_phase(out_zerog, out_sweep, out_freq, out_sample, deconpars);
    fprintf('Scan phase: %5.3f\n', deconpars.field_scan_phase);
    [c_field, phase, lw] = rs_get_field_phase(x_zg,out_zg, ppr_struct);
    fprintf('Center field: %5.3f G. Phase %5.3f. LW %5.4f G.\n', mean(c_field), mean(phase), mean(lw));
    % deconvolve spectra using scan_phase determined earlier
    deconpars.scan_phase_algorithm = 'manual';
    %   deconpars.field_scan_phase = scan_phase;
    deconpars.N_iter = 1;
%     deconpars.data_phase = 0;
    deconpars.data_phase = mean(phase);
    [x_ss,out_y]=rs_sdeconvolve(out_data, out_sweep, out_freq, out_sample, deconpars);
    rec_y = zeros(field_pars.rec.nBins, pars.nP);
    
    % Filter projections
    %   Fcut_off = safeget(rec_info.ppr,'Fcut_off', 0)*1E6;
    %   if Fcut_off > 1e3
    %     fs = raw_info.RSfrequency(1) * Npt;
    %     [B,A] = butter(5,2*Fcut_off/fs);
    %     out_y = filter(B,A,out_y);
    %   end
    
    % Interpolate zero field through all projections
    reference_idx  = find(zero_g);
    zero_field_prj = interp1(reference_idx,c_field,1:pars.nTrace);
    zero_field_prj = field_pars.ppr.data_offset;
    out_y = real(out_y);
    
    deltaB = fbp_struct.MaxGradient*mean(field_pars.rec.size);
    
    x_prime  = (1:nBins)/(nBins-1) * deltaB; x_prime = x_prime - mean(x_prime);
    for ii=1:pars.nP
      rec_y(:,ii) = interp1(x_ss - zero_field_prj, out_y(:,ii), x_prime, 'linear', 0);
    end
    
    %   needs_flip = rs_get_sweep_dir(zero_gradients);
    %   if any(needs_flip)
%     rec_y = flipud(rec_y);
    %   end
    prj_stat(repmat(x_prime', 1,pars.nP),rec_y,{1:pars.nP}, []);
    
  else
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%  4D imaging  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    rec_y = zeros(field_pars.rec.nBins, pars.nP);
    x_ss = zeros(field_pars.rec.nBins, pars.nP);
    
    % Index of projections (no zero gradients)
    ProjectionLayoutIndex = pars.gidx;
    % index that defines zero gradients (==-1) or projections (1...pars.nP)
    zero_g = pars.service_idx == ZEROGRADIENT_INDEX;
    
    % separate data and zero gradient projections
    if any(zero_g)
      out_data  = in_y(:,~zero_g);
      out_zerog = in_y(:,zero_g);
      
      out_freq  = raw_info.RSfrequency(~zero_g);
      out_sweep = raw_info.FieldSweep(~zero_g);
      out_sample = raw_info.sampling(~zero_g);
      UnitSweep = pars.UnitSweep(~zero_g);
    else
      out_data  = in_y;
      out_zerog = [];

      out_freq  = raw_info.RSfrequency;
      out_sweep = raw_info.FieldSweep;
      out_sample = raw_info.sampling;
      UnitSweep = pars.UnitSweep;
    end
    
    % determine image dimensions
    cos_alpha = cos(pars_ext.alpha);
    tan_alpha = max(tan(pars_ext.alpha));
    deltaB =  pars.data.FBP.MaxGradient * field_pars.rec.size / tan_alpha;
    ReconSweep = UnitSweep * deltaB;
    
    IdxShow  = cell(pars.nSpec, 1);
    ScanPhase = cell(pars.nSpec, 1);
    Phase    = cell(pars.nSpec, 1);
    CField   = cell(pars.nSpec, 1);
    LW       = cell(pars.nSpec, 1);
    
    deconpars = ppr_struct;
    range_center = ppr_struct.data_offset;
    range_span = ppr_struct.data_span;

    if any(zero_g)
      for ii=1:pars.nSpec
        data4zerog = out_zerog(:,  [ii, pars.nSpec-ii+1]);
        idxSpec = pars_ext.k == ii;
        RSsweep =  mean(out_sweep(idxSpec));
        RSsample = mean(out_sample(idxSpec));
        RSfrequency = mean(out_freq(idxSpec));
        % use zero gradient traces to determine scan phase
        [x_zg,out_zg, ScanPhase{ii}]=rs_sscan_phase(data4zerog, RSsweep, RSfrequency, RSsample, deconpars);
        fprintf('Scan phase: %5.3f\n', ScanPhase{ii});
        fld_idx = x_zg >= range_center - range_span/2 & x_zg <= range_center + range_span/2;
        [CField{ii}, Phase{ii}, LW{ii}] = rs_get_field_phase(x_zg(fld_idx),out_zg(fld_idx,:), ppr_struct);
        fprintf('Center field: %5.3f G. Phase %5.3f. LW %5.4f G.\n', mean(CField{ii}), mean(Phase{ii}), mean(LW{ii}));
      end
    else
      for ii=1:pars.nSpec
        CField{ii} = range_center;
        Phase{ii} = safeget(deconpars, 'data_phase', 0);
        LW{ii} = 0;
        ScanPhase{ii} = safeget(deconpars, 'field_scan_phase', 0);
        fprintf('Scan phase: %5.3f\n', ScanPhase{ii});
        fprintf('Center field: %5.3f G. Phase %5.3f. LW %5.4f G.\n', mean(CField{ii}), mean(Phase{ii}), mean(LW{ii}));
      end
    end
    
    average_phase = 0;
    average_c_field = 0;
    nav_phase = 0;
    for ii=1:pars.nSpec
      average_phase = average_phase + sum(Phase{ii});
      nav_phase = nav_phase + length(Phase{ii});
      average_c_field = average_c_field + sum(CField{ii});
    end
    average_phase = average_phase / nav_phase;
    %   average_c_field = average_c_field / nav_phase;
    %   average_c_field = average_c_field / nav_phase;
    average_c_field = range_center;
    
    if strcmp(safeget(prc_struct, 'export_prj', 'no'), 'raw')
      dsc.raw.G = [raw_info.GradX, raw_info.GradY, raw_info.GradZ];
      dsc.raw.raw_info = raw_info;
      dsc.raw.raw_info.nTrace = dsc.raw.raw_info.nP;
      deconpars.field_scan_phase = ScanPhase{ii};
      deconpars.scan_phase_algorithm = 'manual';
      %   deconpars.field_scan_phase = scan_phase;
      deconpars.N_iter = 1;
      deconpars.data_phase = average_phase;
      for ii=1:pars.nTrace
        [dsc.raw.B{ii,1},dsc.raw.P{ii,1}]=rs_sdeconvolve(in_y(:,ii), raw_info.FieldSweep(ii), raw_info.RSfrequency(ii), raw_info.sampling(ii), deconpars);
      end
    end
    
    % cycle over different spectral angles
    for ii=1:pars.nSpec
      % select data for particular spectral angle
      idxSpec = pars_ext.k == ii;
      data4sweep  = out_data(:, idxSpec);
      RSfrequency = mean(out_freq(idxSpec));
      RSsweep =  mean(out_sweep(idxSpec));
      RSsample = mean(out_sample(idxSpec));
      image_sweep = mean(ReconSweep(idxSpec));
      cos_sweep = mean(cos_alpha(idxSpec));
      nSpecPrj = numel(find(idxSpec));
      
      % deconvolve spectra using scan_phase determined earlier
      deconpars.field_scan_phase = ScanPhase{ii};
      deconpars.scan_phase_algorithm = 'manual';
      %   deconpars.field_scan_phase = scan_phase;
      deconpars.N_iter = 1;
      deconpars.data_phase = average_phase;
      [x_ss_scan,out_y_scan]=rs_sdeconvolve(data4sweep, RSsweep, RSfrequency, RSsample, deconpars);
      
      % field axis for projections with correct sweep and number of points
      x_ss_sw = linspace(image_sweep/2, -image_sweep/2, field_pars.rec.nBins)+average_c_field;
      
      % interpolate trace to get correct number of points and
      % normalize spectral intensity
      rec_y(:, idxSpec) = interp1(x_ss_scan, out_y_scan, x_ss_sw, 'pchip', 0) / cos_sweep;
      x_ss(:,  idxSpec) = repmat(x_ss_sw', 1, nSpecPrj);
      IdxShow{ii} = idxSpec;
    end
    
    prj_stat(x_ss,rec_y,IdxShow, []);
    
    % Image field axis
    field_pars.rec.deltaH = deltaB;
    
    % compensate for the down-up field scan
    %   rec_y = flip(rec_y, 1);  % MATLAB 2014
    rec_y = flipud(rec_y);
  end
end

if ~rec_enabled
  mat_recFXD = [];
  dsc.prj = rec_y(:, ProjectionLayoutIndex >= 0);
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



