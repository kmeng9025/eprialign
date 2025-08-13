% function [ax, y, dsc] = td_reconstruct(in_y, raw_info, rec_info, ...)
% Pulse EPRI image reconstruction
%   in_opt.phase_algorithm   - algorithm of phase optimization
%        'manual_zero_order' - rotate on angle in_opt.phase_zero_order
%        'max_real_single'   - indepenent, trace by trace
%        'max_real_all'      - use one phase for all slices
%   ...                      - parameter-value comma separated pairs
%   out_pars                 - structure of supplementary information

function [mat_recFXD, rec_info, dsc] = rs_reconstruct(in_y, raw_info, rec_info, fit_info, varargin)

if nargin < 4
  error('Usage: [out_y, out_pars] = td_reconstruct(in_y, in_opt, raw_info, rec_info, vargin)');
elseif nargin > 4
  if ~mod(nargin-1,2)
    for kk=1:2:nargin-1
      in_opt=setfield(in_opt, lower(varargin{kk}), varargin{kk+1});
    end
  else error('td_reconstruct: Wrong amount of arguments')
  end
end

dsc = [];
fbp_struct = raw_info.data.FBP;
ppr_struct = rec_info.ppr;

[pars, pars_ext] = td_GetFBPGradientTable(fbp_struct);

% complain
if pars.nTrace ~= size(in_y, 2), error('Wrong number of projections'); end

% Number of points after folding scans together
Npt   = 2^12;

Nprj = size(in_y, 2);
Sub_points = rec_info.rec.Sub_points;

nH = safeget(ppr_struct, 'baseline_harmonics', 0);
dB = abs(diff(fbp_struct.split_field));

% Spatial image reconstruction
tic
if pars.nSpec == 1 && strcmp(safeget(ppr_struct, 'baseline', 'split_field'), 'split_field')
  ProjectionLayoutIndex = pars.gidx(1:2:end);
  
  rec_info.rec.Size = rec_info.rec.Size*ones(1,3);
  
  % Rapid Scan deconvolution
  [x_ss,out_y]=rs_deconvolve(in_y, raw_info.FieldSweep(1), raw_info.RSfrequency(1), raw_info.sampling(1), Npt);
  %   figure; plot(x_ss,real(out_y));
  
  % Cyclic shift of data to achieve correct scan field phase
  if strcmp(safeget(ppr_struct, 'scan_phase_algorithm', 'manual'), 'auto')
    splitfield_idx = reshape([ProjectionLayoutIndex,ProjectionLayoutIndex]', [1,Nprj]);
    zero_gradients = out_y(:, splitfield_idx == -1);
    
    dPh = rs_scan_phase(zero_gradients);
  else
    dPh = safeget(ppr_struct, 'field_scan_phase', 0);
  end
  np = size(out_y, 1);
  out_y = circshift(out_y, round(np*dPh/360));
  
  [out_yr, BGr]=rs_baseline_sawtooth(x_ss, real(out_y), dB, nH);
  [out_yi, BGi]=rs_baseline_sawtooth(x_ss, imag(out_y), dB, nH);
  out_y = out_yr+1i*out_yi;
  %   BG = BGr+1i*BGi;
  %     figure; plot(x_ss,real(BG));
  %     figure; plot(x_ss,real(out_y));
  %     figure; plot(x_ss,imag(out_y));
  
  ProjectionLayoutIndex = pars.gidx(1:2:end);
  
  Nprj = size(out_y, 2);
  rec_y = zeros(Sub_points, Nprj);
  
  
  % Baseline offset correction
  bl_idx = [1:100,Npt/2-100:Npt/2];
  OFF = mean(out_y(bl_idx,:));
  out_y = out_y - repmat(OFF, [Npt, 1]);
  
  % Filter projections
  Fcut_off = safeget(rec_info.ppr,'Fcut_off', 0)*1E6;
  if Fcut_off > 1e3
    fs = raw_info.RSfrequency(1) * Npt;
    [B,A] = butter(5,2*Fcut_off/fs);
    out_y = filter(B,A,out_y);
  end
  
  % Select data part for reconstruction
  % The structure of data expected is _______/\____|____/\_______
  if strcmp(safeget(ppr_struct, 'baseline', 'split_field'), 'split_field')
    x_ss = x_ss(1:Npt/2);
    out_y = out_y(1:Npt/2,:);
  else
    zero_gradients = out_y(:, ProjectionLayoutIndex == -1);
    nZG = size(zero_gradients, 2);
    
    c_shifts = zeros(nZG, 1);
    
    for ii=1:size(zero_gradients, 2)
      c_shifts(ii) = rs_get_zero_field(x_ss, zero_gradients(:,ii));
    end
    c_shift = fix(mean(c_shifts));
    
    [out_ys(:,1,:), out_ys(:,2,:)] = rs_split_trace(out_y, c_shift);
    out_y = squeeze(sum(out_ys, 2));
  end
  
  % Phase the data
  zero_gradients = out_y(:, ProjectionLayoutIndex == -1);
  nZG = size(zero_gradients, 2);
  zero_fields_fit   = zeros(nZG, 1);
  if strcmp(safeget(ppr_struct, 'phase_algorithm', 'auto'), 'auto')
    phases = zeros(nZG, 1);
    
    for ii=1:size(zero_gradients, 2)
      %     [tr1, tr2] = rs_split_trace(zero_gradients(:,ii), c_shift, isflip);
      [phases(ii), zero_fields_fit(ii)] = rs_get_phase(x_ss, zero_gradients(:,ii), fit_info);
    end
    the_phase = mean(phases);
    disp(sprintf('EPR phase: %4.2f degree', the_phase));
  else
    the_phase = safeget(ppr_struct, 'data_phase', 0);
    zero_gradients = zero_gradients *exp(-1i*the_phase*pi/180);
    for ii=1:size(zero_gradients, 2)
      [m,fidx] = max(zero_gradients(:,ii));
      zero_fields_fit(ii) = x_ss(fidx);
    end
  end
  out_y = out_y *exp(-1i*the_phase*pi/180);
  figure; plot(1:2048, real( out_y(:,1)), 1:2048, imag( out_y(:,1)))
  
  out_y = real(out_y);
  
  % Interpolate zero field through all priojections
  reference_idx  = find(ProjectionLayoutIndex == -1);
  zero_field_prj = interp1(reference_idx,zero_fields_fit(:,1),1:Nprj);
  deltaB = fbp_struct.MaxGradient*mean(rec_info.rec.Size);
  
  x_prime  = (1:Sub_points)/(Sub_points-1) * deltaB; x_prime = x_prime - mean(x_prime);
  for ii=1:Nprj
    rec_y(:,ii) = interp1(x_ss - zero_field_prj(ii), fix_baseline(out_y(:,ii)), x_prime, 'pchip', 0);
    %     figure(100); clf
    %     subplot(2,1,1); hold on;
    %     iplot(x_ss - zero_field_prj(ii,1), out_y(:,ii));
    %     ax = axis; plot(deltaB/2*[-1,1], ax(3)/2*[1,1], 'r'); axis tight
    %     subplot(2,1,2); hold on;
    %     iplot(x_prime, rec_y(:,ii));
    %     axis tight
    % %     axis([-inf,inf,0,max(rec_y(:,ii))]);
    %     ax=axis; plot([0,0], ax([3,4]), 'r');
    %     pause
  end
  

  %   needs_flip = rs_get_sweep_dir(zero_gradients);
  %   if any(needs_flip)
  rec_y = flipud(rec_y);
  %   end
else
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%  4D imaging  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  out_y = zeros(Npt, Nprj);
  
  % deconvolve spectra
  x_ss = zeros(Npt, Nprj);
  for ii=1:Nprj
    [x_ss(:,ii),out_y(:,ii)]=rs_deconvolve(in_y(:,ii), raw_info.FieldSweep(ii), raw_info.RSfrequency(ii), raw_info.sampling(ii), Npt);
  end
  
  % Filter projections
  Fcut_off = safeget(rec_info.ppr,'Fcut_off', 0)*1E6;
  if Fcut_off > 1e3
    fs = raw_info.RSfrequency(1) * Npt;
    [B,A] = butter(5,2*Fcut_off/fs);
    out_y = filter(B,A,out_y);
  end
  
  % cw reconstruction
  deltaH = fbp_struct.MaxGradient*rec_info.rec.Size / abs(tan(pi/fbp_struct.nSpec/2-pi/2));
  rec_info.rec.deltaH = deltaH*sqrt(2);
  
  % zero-gradients projections
  grad_idx = pars.gidx(:) == -1;
  x_scan = x_ss(:,grad_idx);
  zero_gradients = out_y(:, grad_idx);
  np = size(zero_gradients, 1);
  nG = size(zero_gradients, 2);
  dPh = zeros(nG/2,1);
  Ph = zeros(nG/2,1);
  zero_fields_fit = zeros(nG/2,1);
  fit_info4phase = struct('spin_probe', 'Lorentzian');
  for ii=1:nG/2
    tr_idx = (ii-1)*2+[1, 2];
    dPh(ii) = rs_scan_phase(zero_gradients(:,tr_idx));
    shifted_array = circshift(zero_gradients(:,tr_idx),  round(np*dPh(ii)/360));
    zero_g = rs_baseline_sawtooth(x_scan(:,tr_idx(1)), real(shifted_array), dB, nH)+...
      1i*rs_baseline_sawtooth(x_scan(:,tr_idx(1)), imag(shifted_array), dB, nH);
    
    x_single_scan   = x_scan(1:np/2,tr_idx(1));
    [Ph(ii), zero_fields_fit(ii)] = rs_get_phase(x_single_scan, zero_g(1:np/2), fit_info4phase);
  end
  flipdata = true;
  disp('Scan [deg], Phase [deg], X0 [G]')
  disp([dPh, Ph, zero_fields_fit])
  
  % average for beginning and the end of scan
  scanPh = (dPh(1:pars.nSpec/2) +  flipud(dPh(pars.nSpec/2+1:pars.nSpec))) / 2;
  % append for missing spectral angles with the same sweep
  scanPh(pars.nSpec/2+1:pars.nSpec) = flipud(scanPh(1:pars.nSpec/2));
  
  zero_field = mean(zero_fields_fit);
  
  ProjectionLayoutIndex = pars.gidx(~grad_idx);
  ProjectionLayoutIndex = ProjectionLayoutIndex(1:2:end);
  
  % extract projections projections
  x_ss = x_ss(:, ~grad_idx);
  x_ss = reshape(x_ss, [size(x_ss, 1), 2, size(x_ss, 2)/2]);
  out_y = out_y(:, ~grad_idx);
  out_y = reshape(out_y, [size(out_y, 1), 2, size(out_y, 2)/2]);
  
  RSfrequency = raw_info.RSfrequency(~grad_idx);
  RSfrequency = RSfrequency(1:2:end);
  UnitSweep = pars.UnitSweep(~grad_idx);
  UnitSweep = UnitSweep(1:2:end);
  
  tan_alpha = max(tan(pars_ext.alpha));
  deltaB =  pars.data.FBP.MaxGradient * rec_info.rec.Size / tan_alpha;
  ReconSweep = UnitSweep * deltaB;
  
  for ii=1:pars.nSpec
    swp_idx   = pars_ext.k == ii;
    % Cyclic shift of data to achieve correct scan field phase
    if strcmp(safeget(ppr_struct, 'scan_phase_algorithm', 'manual'), 'auto')
      scanPhase = scanPh(ii);
    else
      scanPhase = safeget(ppr_struct, 'field_scan_phase', 0);
    end
    np = size(out_y, 1);
    sawtooth_array = circshift(out_y(:,:,swp_idx),  round(np*scanPhase/360));
    cos_alpha = mean(cos(pars_ext.alpha(swp_idx)));
    
    sz = size(sawtooth_array);
    sawtooth_array = reshape(sawtooth_array, sz(1), sz(2)*sz(3));
    x_ss_sw = x_ss(:,1,swp_idx);
    x_scan = x_ss_sw(:,1,1);
    ReconSweep_scan = mean(ReconSweep(swp_idx));
    [out_yr, BGr]=rs_baseline_sawtooth(x_scan, real(sawtooth_array), dB, nH);
    %       [out_yi, BGi]=rs_baseline_sawtooth(x_scan, imag(sawtooth_array), dB, nH);
    out_yi = zeros(size(out_yr));
    out_y_bl = out_yr(1:sz(1)/2,:)+1i*out_yi(1:sz(1)/2,:);
    
    %     BG = BGr+1i*BGi;
    %     figure; plot(x_scan,real(BG));
    %     figure; plot(x_scan,real(out_y));
    %     figure; plot(x_scan,imag(out_y));
    
    % Baseline offset correction
    %     Npt = size(out_y_bl, 1);
    %     bl_idx = [1:100,Npt-100:Npt];
    %     OFF = mean(out_y_bl(bl_idx,:));
    %     out_y_bl = out_y_bl - repmat(OFF, [Npt, 1]);
    
    %     % Filter projections
    %     Fcut_off = safeget(rec_info.ppr,'Fcut_off', 0)*1E6;
    %     if Fcut_off > 1e3
    %       fs = RSfrequency * Npt;
    %       [B,A] = butter(5,2*Fcut_off/fs);
    %       out_y = filter(B,A,out_y);
    %     end
    
    out_y_bl = real(out_y_bl);
    
    x_prime  = (1:Sub_points)/(Sub_points-1) * ReconSweep_scan; x_prime = x_prime' - mean(x_prime);
    int_y = zeros(Sub_points, sz(3));
    x_single_scan   = x_scan(1:np/2);
    
    for jj=1:length(find(swp_idx))
      int_y(:,jj) = fix_baseline(interp1(x_single_scan - zero_field, fix_baseline(out_y_bl(:,jj)), x_prime, 'pchip', 0));
    end
    
    rec_y(:,swp_idx) = int_y / cos_alpha;
  end
  
  if flipdata
    rec_y = flipud(rec_y);
  end  
end

toc

rec_enabled = isequal(safeget(rec_info.rec, 'Enabled','on'), 'on');
if ~rec_enabled
  mat_recFXD = [];
  dsc.prj = rec_y(:, ProjectionLayoutIndex >= 0);
  return;
end

sz = size(rec_y);
% remove baseline traces
mat = zeros([sz(1), prod(pars.Dim)]);
rem_idx = ProjectionLayoutIndex >= 0;
mat(:, ProjectionLayoutIndex(rem_idx)) = real(single(rec_y(:,rem_idx)));
mat_out = reshape(mat, [sz(1), pars.Dim]);

% -------------------------------------------------------------------------
% -------------- R E C O N S T R U C T I O N ----------------------------
% -------------------------------------------------------------------------

com = cw_set_com(raw_info, [], pars.nSpec);

% sub/supersampling
% switch safeget(rec, 'SubSampling', 'gaussian')
%   case 'imresize'
%     mat_out = reshape(yy, [Npt, prod(pars.Dim)]);
%     mat_out = imresize(mat_out, [Sub_points, prod(pars.Dim)]);
%   case 'interpft'
%     mat_out = interpft(yyy, Sub_points, 1);
%   case 'spline'
%     yyy = reshape(yy, [Npt,prod(sz(2:end))]);
%     for ii = 1 : prod(sz(2:end))
%       mat_out(:, ii) = spline(1:sz(1), yyy(:, ii), linspace(1, sz(1), Sub_points));
%     end
%   case 'gaussian' % % gaussian convolution
%     mat_out=subsampleimagedata(mat_out, Sub_points, Npt/rec.Sub_points, 1);
% end
mat_out = reshape(mat_out, [Sub_points,fbp_struct.nSpec,fbp_struct.nAz,fbp_struct.nPolar]);

% Interpolating
switch safeget(raw_info.data.FBP,'angle_sampling','uniform_spatial')
  case {'uniform_spatial','uniform_spatial_flip'}
    mat_out=InterpToUniformAngle(mat_out,'imgData');
end

% mat_out = y;
if strcmp(rec_enabled, 'off'); mat_recFXD=[]; return; end

% Reconstruction
MatrixGUI(mat_out)

% mat_recFXD = Recon_AI_Filt_BP(mat_out,com);
radon_pars.ELA =  raw_info.data.FBP;
recon_pars = rec_info.rec;
radon_pars.size = deltaB;
radon_pars.nBins = rec_info.rec.Sub_points;
mat_recFXD = iradon_d2d_mstage(mat_out, radon_pars, recon_pars);

% normalize amplitude on the unit volume/1D
switch 14
  case 1, n = 4;
  case 14, n = 3; % 3D experiment
end

% Set software scaling factor
rec_info.ampSS = 1 ./ (rec_info.rec.Size(1)*0.01/Sub_points)^(n-1); % meters ;)

if safeget(rec_info.rec, 'DoublePoints', 0)
  switch n
    case 3
      mat_recFXD = reshape(mat_recFXD, [2,Sub_points,2,Sub_points,2,Sub_points]);
      mat_recFXD =  squeeze(sum(sum(sum(mat_recFXD, 1), 3), 5))/8;
  end
end

disp('    Reconstruction is finished.');
rec_info.com = com;

function y1 = fix_baseline(y)
% option 1: do nothing
% y1 = y;
%option 2: simmetrize using derivative
deriv = diff(y);
OFF = mean(deriv); deriv = deriv - OFF;
intg = cumsum(deriv); y1 = [intg; intg(end)];
% option 3 linear correction




