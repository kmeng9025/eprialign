% function [ax, y, dsc] = td_preprocess(mat_int, mat_info, mat_rec_info)
% mat_int - baseline corrected data
% mat_info - structure of loading parameters
% mat_rec_info - structure of reconstruction parameters

% Pulse EPRI image reconstruction
%   in_opt.phase_algorithm   - algorithm of phase optimization
%        'manual_zero_order' - rotate on angle in_opt.phase_zero_order
%        'max_real_single'   - indepenent, trace by trace
%        'max_real_all'      - use one phase for all slices
%   ...                      - parameter-value comma separated pairs
%   out_pars                 - structure of supplementary information

function [rx, yyyy, par_fields, dsc] = epri_ese_preprocess(raw_projection_data, mat_info, par_fields, other)

st_fft = par_fields.fft;
fbp_struct = mat_info.data.FBP;

td_baseline_display   = safeget(par_fields.td, 'baseline_display','off');
td_phase_display      = safeget(par_fields.td, 'phase_display','off');
fft_fft_display     = safeget(st_fft, 'fft_display','off');

% -------------------------------------------------------------------------
% ---------- T I M E   D O M A I N ----------------------------------------
% -------------------------------------------------------------------------

% time axis
t = mat_info.t_ax;

% clip data trace
clipat = safeget(par_fields.td, 'data_clip_ns', 10000.0);
last_point = numel(find(t < clipat*1e-9));
if last_point ~= size(raw_projection_data,1)
    raw_projection_data = raw_projection_data(1:last_point, :, :);
    t = t(1:last_point);
    if isfield(other, 'zerog')
        other.zerog = other.zerog(1:last_point, :, :);
    end
end

% baseline correction and baseline glitches removal
[y,st_td] = kv_baseline_correction(raw_projection_data, par_fields.td);
dsc.baseline_out = st_td;
if isfield(other, 'zerog')
  y_zerog = kv_baseline_correction(other.zerog, par_fields.td);
end

% suppress signals during dead time
% acq_window_start declares offset of the start of time to acquisition
% relative to echo position
dead_time = safeget(st_td,'dead_time',0);
acq_window_start = safeget(st_td,'acq_window_start',0);
sz_src = [size(y),1];
first_idx = zeros(sz_src(2), 1);
if length(mat_info.tau2) == 1 && sz_src(2) > 1
  mat_info.tau2 = mat_info.tau2*(1:sz_src(2));
end
for ii=1:sz_src(2)
  idx = t+acq_window_start*1E-9+mat_info.tau2(ii)/2 <  dead_time*1E-9;
  kkk = find(idx,1,'last');
  if ~isempty(kkk)
    y(idx,ii,:) = 0;
    first_idx(ii) = kkk;
  else
    first_idx(ii) = 1;
  end
end

% display baseline stats
switch td_baseline_display
  case 'all'
    baseline_stat(mat_info.t_ax, y, st_td)
  case 'all_by_slice'
    figure(1); clf;
    for ii=1:size(y,2), plot(1:length(t),real(y(:,ii)),1:length(t),imag(y(:,ii))); pause; end
end

BW = 7E6;
yfiltered = zeros(size(y));
for ii=1:size(y,2)
  for jj=1:size(y,3)
    yfiltered(:,ii,jj) = epr_lowpass(mean(diff(t)), y(:,ii,jj).*apowin('ham+', size(y,1)), BW);
  end
end

% determine the echo zero time
try
  [par_fields.td.max_idx, dsc.echo]=ese_getzerotime(yfiltered, par_fields.td);
catch err
  if ~strcmp(td_baseline_display, 'all')
    baseline_stat(mat_info.ax, yfiltered, st_td)
  end
  disp(err);
end

% determine the echo phase
[par_fields.td.right_phase, dsc.phase]=ese_getphase(y, par_fields.td);
par_fields.td.is_performed = ~isempty(par_fields.td.right_phase);

if par_fields.td.is_performed
  yy = zeros(sz_src);
  for ii=1:sz_src(2)
    for jj=1:sz_src(3)
      for kk=1:sz_src(4)
        yy(:, ii, jj, kk) = phase_zero_order(y(:, ii, jj, kk), par_fields.td.right_phase(ii,jj));
      end
    end
  end
end
if isfield(other, 'zerog')
  for ii=1:size(y_zerog,2)
    for jj=1:size(y_zerog,3)
      for kk=1:size(y_zerog,4)
        yy_zerog(:, ii, jj, kk) = phase_zero_order(y_zerog(:, ii, jj, kk), mean(par_fields.td.right_phase(ii,:)));
      end
    end
  end
end


% Display phase stats
switch td_phase_display
  case 'all'
    phase_stat(t, yy, par_fields.td, st_fft, dsc);
  case 'all_by_slice'
    figure(2); clf; axis
    xlabel('Points'); title('After phase correction, maximum is marked');
    mx = max(max(real(y)));
    if length(st_td.max_idx)==1
      for ii=1:size(y,2), plot(t,real(y(:,ii)),t,imag(y(:,ii)),t(st_td.max_idx)*[1,1],[-mx,mx]);
        axis('tight');
        text(.9,.9,num2str(ii),'Units','Normalized','Position',[0.9,0.9,0]); pause;
      end
    else
      for ii=1:size(y,2), plot(t,real(y(:,ii)),t,imag(y(:,ii)),t(st_td.max_idx(ii))*[1,1],[-mx,mx]);
        axis('tight');
        text(.9,.9,num2str(ii),'Units','Normalized','Position',[0.9,0.9,0]); pause;
      end
      axes('position',[.75,.75,.15,.15]);
      hist(st_td.max_idx)
    end
    phase_stat(in_ax, y, st_td, st_fft);
end

% -------------------------------------------------------------------------
% ---- F O U R I E R   T R A N S F O R M A T I O N ------------------------
% -------------------------------------------------------------------------

yyy = reshape(yy, sz_src(1), prod(sz_src(2:end)));
offset = safeget(st_fft,'xshift',0);

dx = mean(diff(mat_info.t_ax));
switch st_fft.data
  case '_0_'
    % correct only for mat_rec_info.td.max_idx < 1/2 of the trace length
    zf = 2^(floor(log2(2*(sz_src(1)-par_fields.td.max_idx))+0.5)+1);
    yyy_zf = zeros([zf, sz_src(2),sz_src(3)]);
    idx_off = zf/2 - par_fields.td.max_idx + 1;
    %     yy(mat_rec_info.td.max_idx,:,:) = yy(mat_rec_info.td.max_idx,:,:)*1.4;
    % filling with data
    yyy_zf(idx_off + (1:sz_src(1)),:,:)=yy(:,:,:);
    % filling with flipped data
    for ii=1:sz_src(2)
      last_point = idx_off + first_idx(ii) + 2; % two points to awoid ovelap problems
      yyy_zf(2:last_point+1,ii,:)=real(yyy_zf(end:-1:end-last_point+1,ii,:))-...
        1i*imag(yyy_zf(end:-1:end-last_point+1,ii,:));
      %       yyy_zf(last_point,:,:) = yyy_zf(last_point,:,:)*1.4;
    end
    yyy_zf = reshape(yyy_zf, zf, prod(sz_src(2:end)));
    % fft
    yyyy = fftshift(ifft(fftshift(yyy_zf, 1)), 1);
    rx = (-zf/2:zf/2-1).'*1/dx/zf*1E-6 + offset;
  case {'0_', '0__0'}
    % zero gradient processing
    if strcmp(safeget(st_fft, 'xshift_mode', 'none'), 'fit_gzero_trace')
        zerog = other.zerog(par_fields.td.max_idx:end,end);
        zf = 2^(floor(log2(size(yyy,1))+0.5)+4);
        zerog(zf,:) = 0;
        ref_spec= fftshift(ifft(zerog), 1);
        rx = (-zf/2:zf/2-1).'*1/dx/zf*1E-6;
        [~,maxidx] = max(abs(ref_spec));
        st_fft.offset = rx(maxidx);
        st_fft.profile_center_offset = st_fft.offset;
        offset = st_fft.offset;
%         figure(1); plot(rx, real(ref_spec), rx,abs(ref_spec)); axis([-5,5,-Inf, +Inf])
    end

    yyy = yyy(par_fields.td.max_idx:end,:);
    % zero padding
    zf = 2^(floor(log2(size(yyy,1))+0.5)+1); 
%     zf = 2048;
    yyy(zf,:) = 0;
    % fft
    if strcmp(st_fft.data, '0__0')
      % add conjugated part to the end
      yyy(zf*2,:) = 0;
      yyy(zf + 2 : zf*2, :) = conj(flip(yyy(2 : zf, :), 1));
%       yyy = yyy / 2; % addition of conjugated doubled the intensity
      yyyy = fftshift(fft(conj(yyy)), 1);
      rx = (-zf:zf-1).'*1/dx/zf/2*1E-6 + offset;
      
      if isfield(other, 'zerog')
        sz_gsrc = size(yy_zerog);
        yy_zerog = reshape(y_zerog, sz_src(1), prod(sz_gsrc(2:end)));
        yyy_zerog = yy_zerog(par_fields.td.max_idx:end,:);
        yyy_zerog(zf*2,:) = 0;
        yyy_zerog(zf + 2 : zf*2, :) = flip(yyy_zerog(2 : zf, :), 1);
        yyyy_zerog = fftshift(ifft(yyy_zerog), 1);
      end
      
      if 1==2 % understanding the advantages of 
        sz_gsrc = size(yy_zerog);
        yy_zerog = reshape(y_zerog, sz_src(1), prod(sz_gsrc(2:end)));
        yyy_zerog = yy_zerog(par_fields.td.max_idx:end,:);
        yyy_zerog(zf,:) = 0;
        
        figure(100); clf;
        subplot(2,1,1); hold on
        plot(real(yyy_zerog(:,8))); plot(imag(yyy_zerog(:,8)));
        cyyy = yyy_zerog(:,8);
        cyyy(zf/2 + 2 : zf, :) = conj(flip(cyyy(2 : zf/2, :), 1));
        cyyy = cyyy / 2;
        plot(real(cyyy)); plot(imag(cyyy));
       
        subplot(2,1,2); hold on
        idx = 1024 + (-100:100);
        yyyy_zerog = fftshift(fft(yyy_zerog(:,8)), 1);
        plot(idx, real(yyyy_zerog(idx)),idx,imag(yyyy_zerog(idx)));
        
        cyyyy = fftshift(ifft(cyyy), 1);
        plot(idx, real(cyyyy(idx)),idx,imag(cyyyy(idx)));
        legend({'r','i','simr','simi'})
        axis tight
       
      end
    else
      yyyy = fftshift(ifft(yyy), 1);
      rx = (-zf/2:zf/2-1).'*1/dx/zf*1E-6 + offset;
      
      if isfield(other, 'zerog')
        sz_gsrc = size(yy_zerog);
        yy_zerog = reshape(y_zerog, sz_src(1), prod(sz_gsrc(2:end)));
        yyy_zerog = yy_zerog(par_fields.td.max_idx:end,:);
        yyy_zerog(zf,:) = 0;
        yyyy_zerog = fftshift(ifft(yyy_zerog), 1);
      end
    end
end

% Post FFT filtering
% switch safeget(st_fft, 'filt', 'none')
%   case 'savgol'
%     ry = smooth(ry, st_fft.tc, st_fft.filt);
% end

other = [];

par_fields.MB0 = safeget(par_fields, 'MB0', []);
trunc_lim = safeget(par_fields.MB0, 'MBProjTrunc', [-8 8]); %MHz

% FOVidx
if isfield(st_fft, 'FOV')
  FOVidx = abs(rx) < st_fft.FOV/2;
end

profile_threshold = safeget(st_fft, 'profile_threshold', 0.05);
pars.Q = str2double(safeget(mat_info.Unformatted, 'sample_info_Q', '20'));
pars.Resonator = safeget(mat_info.Unformatted, 'sample_info_resonator', 'LGR19L');
profile = epri_cavity_profile(rx, st_fft, pars);
profile(~FOVidx) = profile_threshold;

switch safeget(fbp_struct, 'scheme', 'single_b')
  case 'single_b'
    yyyy = yyyy./profile(:, ones(size(yyyy,2),1));
    % use offset parameter
    RO = safeget(mat_info, 'Offset', [])*2.8025 * 0.0;
    if any(RO > 0.00001)
      for ii=1:size(yyyy,2)
        yyyy(:,ii) = interp1(rx, yyyy(:,ii), rx-RO(jj), 'linear', 0);
      end
    end
    if isfield(other, 'zerog')
      yyyy_zerog = yyyy_zerog./profile(:, ones(size(yyyy_zerog,2),1));
      % use offset parameter
      RO = safeget(mat_info, 'Offset', [])*2.8025* 0.0;
      if any(RO > 0.00001)
        for ii=1:size(yyyy,2)
          yyyy(:,ii) = interp1(rx, yyyy(:,ii), rx-RO(jj), 'linear', 0);
        end
      end
    end
  case 'multi_b'
    yyyy = reshape(yyyy, size(yyyy,1), sz_src(2), sz_src(3), size(y,4));
    yyyy(abs(rx)<trunc_lim(1),:,:,:,:)=0; yyyy(abs(rx)>trunc_lim(2),:,:,:,:)=0;
    yyyy = mb_shiftCorrection(yyyy, mat_info, rx);
    ZFSind = find(mat_info.Boffset==0);
    mat_info.frequency = rx;
    [yyyy, mat_info] = mb_recombineProjections(yyyy, mat_info, par_fields, profile, ZFSind);
    sz_src(3) = mat_info.nP;
end

yyyy = reshape(yyyy, [size(yyyy,1), sz_src(2), sz_src(3)]);

% Export projections in k-space
dsc.k = dx*(1:size(yyyy,1))*par_fields.td.t2k; dsc.k = dsc.k - mean(dsc.k);
dsc.kproj   = fftshift(fft(fftshift(yyyy), [], 1));
kidx = abs(dsc.k) <= par_fields.td.kmax;
dsc.k = dsc.k(kidx);
dsc.kproj = dsc.kproj(kidx,:,:);

% Limit data to FOV
if isfield(st_fft, 'FOV')
  yyyy = yyyy(FOVidx,:,:);
  rx   = rx(FOVidx);
  
  if isfield(other, 'zerog')
    yyyy_zerog = reshape(yyyy_zerog, [size(yyyy_zerog,1), sz_src(2),2]);
    dsc.zerog = yyyy_zerog(FOVidx,:,:);
  end
end

% switch safeget(st_fft, 'clearence_correction', 'none')
%   case 'exponential'
%   case 'polynomial'
%     correction_order = safeget(st_fft, 'clearence_correction_order', 3);
%     p = polyfit(1:rsz(2),dsc.fft_intg,correction_order);
%     fit_integral = polyval(p,1:rsz(2));
%     profile = fit_integral/max(fit_integral);
%     profile(profile < profile_threshold) = profile_threshold;
%     ry = ry./profile(ones(rsz(1),1), :);
% end

% Export clearence data
for ii = 1:sz_src(2)
  dsc.export_clearence(:,ii) = squeeze(trapz(real(yyyy(:,ii,:)),1));
end

switch fft_fft_display
  case 'all'
    fft_stat(rx, yyyy, st_fft);
  case 'all_by_slice'
    figure(4); clf;
    for ii=1:size(yyyy,3), plot(rx,real(yyyy(:,1,ii)),'b-',rx,imag(yyyy(:,1,ii)),'g-'); axis('tight'); grid('on');
      text(.9,.9,num2str(ii),'Units','Normalized','Position',[0.9,0.9,0]); pause;
    end
    fft_stat(rx, yyyy, st_fft);
end

%---------------------------------------------------------------
function baseline_stat(in_ax, y, st_td)
figure(1); clf;
sz = size(y);
pst = epr_CalcAxesPos(sz(2), 1, [0.06 0.0005], [0.04 0.05]);

for tau = 1:sz(2)
  h(tau) = axes('Position', pst(tau,:)); hold on
  mxr = max(max(real(y(:,tau,:)))); mnr = min(min(real(y(:,tau,:))));
  mxi = max(max(imag(y(:,tau,:)))); mni = min(min(imag(y(:,tau,:))));
  for ii=1:size(y,3), plot(1:size(y,1),real(y(:,tau,ii)),1:size(y,1),imag(y(:,tau,ii))); end
  %if st_td.is_performed
  plot(st_td.baseline_area, min(mnr,mni)*ones(size(st_td.baseline_area,1)),'m*');
  %end
  if tau == 1, title('After base line correction'); end
  axis('tight'); xlabel('Points');
  % if st_td.is_performed
  axes('position',pst(tau,:) + [pst(tau,3)/2, pst(tau,4)*3/4, -pst(tau,3)*.8, -pst(tau,4)*.8]); hold on
  plot(1:sz(3), real(st_td.baseline_zero(:,tau)),1:sz(3), imag(st_td.baseline_zero(:,tau)));
  axis('tight'); title('Shift');
  axes('position',pst(tau,:) + [pst(tau,3)*3/4, pst(tau,4)*3/4, -pst(tau,3)*.8, -pst(tau,4)*.8]); hold on
  ssum = squeeze(sum(y(st_td.baseline_area,tau,:)));
  plot(1:sz(3), real(ssum), 1:sz(3), imag(ssum));
  axis('tight'); title('Sum');
end

%---------------------------------------------------------------
function phase_stat(t, y, st_td, st_fft, dsc)

sz   = size(y);
x_dim = 1:length(t);

figure(safeget(st_td, 'FigPhase', 2)); clf;
% apodization window profile and window-corrected traces
% fft1 = st_fft; fft1.fft = 0;
% if strcmp(fft1.data, '_0_')
%     fft1.center = st_td.max_idx(1);
%     fft1.lshift = 0;
% else
%     fft1.lshift = st_td.max_idx(1) - 1;
%     fft1.center = 0;
% end
% [rax, ry, awin] = kv_fft(struct('x', t), y, fft1);
% mx = max(real(ry(:))); mn = min(real(ry(:)));
% mx = max(max(imag(ry(:))),mx); mn = 1.5*min(min(imag(ry(:))),mn);

% create area of visible and invisible signal
idx = find(x_dim >= st_td.max_idx);
iidx = find(x_dim <= st_td.max_idx);

% for ii=1:sz2, plot(x_dim(idx),real(ry(idx,ii)),...
%     x_dim(idx),imag(ry(idx,ii)),st_td.max_idx_all(ii)*[1,1],[mn,mx]); end

pst=epr_CalcAxesPos(sz(2), 1, [0.06 0.0005], [0.04 0.05]);

h = zeros(sz(2), 1);
for ii = 1:sz(2)
  h(ii) = axes('Position', pst(ii,:)); hold on
  for jj=1:sz(3)
    plot(t(idx), real(y(idx,ii,jj)), 'b');
    plot(t(idx),imag(y(idx,ii,jj)), 'g');
    plot(t(iidx), real(y(iidx,ii,jj)), ...
      t(iidx),imag(y(iidx,ii,jj)), 'Color', [0.7, 0.7, 0.7]);
  end
  axis tight
  YLim = get(h(ii), 'YLim');
  plot(t(st_td.max_idx)*[1,1],YLim,'m','LineWidth',2);
  if st_td.is_performed && ~(any(pst(ii,:) + [0.55, 0.025, -0.58, -0.05] < 0))
    axes('position',pst(ii,:) + [0.55, 0.025, -0.58, -0.05]); hold on
    plot(1:size(dsc.phase.phase,2), dsc.phase.phase(ii,:)*180/pi);
    plot(1:size(st_td.right_phase,2), st_td.right_phase(ii,:)*180/pi, 'r');
    axis tight
    %     hhh = legend({'est.','used'}, 'Box', 'off');
  end
end
set(h(1:end-1), 'XTickLabel', '');
set(h, 'Box', 'on');
title(h(1), sprintf('idx = %i/%i (%4.1fns)', st_td.max_idx, sz(1), t(st_td.max_idx)*1e9))
xlabel(h(end), 'Points')

% plot(mx*awin, 'k');
% axis('tight'); xlabel('Points'); title('After phase correction, maximum is marked');
% if st_td.is_performed
%     % Time zero plot
%     h = axes('position',[.63,.16,.12,.12]);
%     hist(st_td.max_idx_all); axis('tight');
%     title('time 0');
%     % zero time plot
%     h = axes('position',[.78,.16,.12,.12]);
%     plot(st_td.max_idx_all); hold on;
%     if length(st_td.max_idx) == 1
%       plot(st_td.max_idx*ones(sz2,1), 'm')
%     else
%       plot(st_td.max_idx, 'm')
%     end
%     title(['time 0, ',num2str(st_td.max_idx)]); axis('tight');
%     % echo amplitude plot
%     h = axes('position',[.78,.75,.12,.12]);
%     if length(st_td.max_idx)==1,
%         sel = st_td.max_idx;
%         plot(sum(abs(y(sel,:)),1));
%     else
%         sel = st_td.max_idx;
%         plot(sum(abs(y(sel,:)),1));
%     end
%     title('Echo amplitude');
%     % Phase plot
%     axis([-Inf,Inf,0,Inf]);
%     h = axes('position',[.63,.75,.12,.12]);
%     plot(st_td.phase_zero_order_all*180/pi); hold on
%     if length(st_td.phase_zero_order) == 1
%       plot(st_td.phase_zero_order*180/pi*ones(sz2,1), 'm')
%     else
%       plot(st_td.phase_zero_order*180/pi, 'm')
%     end
%     title('Phase'); axis('tight');
% end

function fft_stat(rx, ry, st_fft)
figure(safeget(st_fft, 'FigFFT', 4)); clf; hold on

sz = size(ry);
pst=epr_CalcAxesPos(sz(2), 1, [0.06 0.0005], [0.04 0.05]);

for ii = 1:sz(2)
  h(ii) = axes('Position', pst(ii,:)); hold on
  for jj=1:sz(3)
    plot(rx, real(ry(:,ii,jj)), 'b');
    %     plot(rx,imag(ry(:,ii,jj)), 'g');
  end
  axis tight
%   hh(ii) = axes('position',pst(ii,:) + [0.65, 0.1, -0.67, -0.11]); hold on
  hh(ii) = axes('position',max(pst(ii,:) + [0.65, 0.1, -0.67, -0.11], zeros(1,4)+0.02)); hold on
  sum_ry = squeeze(trapz(real(ry(:,ii,:)),1));
  plot(sum_ry); axis('tight'); title('Integral');
  text(0.25, 0.5, sprintf('Avg: %f', mean(sum_ry)),'Units','Normalized')
  axis([-Inf, Inf, 0, Inf])
end
set(h(1:end-1), 'XTickLabel', '');
set(h, 'Box', 'on');
set(hh, 'Box', 'on');
xlabel(h(end), '[MHz]')

if strcmp(safeget(st_fft, 'fft_display_imag', 'none'), 'all')
  figure(safeget(st_fft, 'FigFFT', 5)); clf; hold on
  
  sz = size(ry);
  pst=epr_CalcAxesPos(sz(2), 1, [0.06 0.0005], [0.04 0.05]);
  
  for ii = 1:sz(2)
    h(ii) = axes('Position', pst(ii,:)); hold on
    for jj=1:sz(3)
      plot(rx, imag(ry(:,ii,jj)), 'b');
      %     plot(rx,imag(ry(:,ii,jj)), 'g');
    end
    axis tight
    hh(ii) = axes('position',pst(ii,:) + [0.65, 0.1, -0.67, -0.11]); hold on
    sum_ry = squeeze(trapz(imag(ry(:,ii,:)),1));
    plot(sum_ry); axis('tight'); title('Integral');
    axis([-Inf, Inf, 0, Inf])
  end
  set(h(1:end-1), 'XTickLabel', '');
  set(h, 'Box', 'on');
  set(hh, 'Box', 'on');
  xlabel(h(end), 'MHz')
end

% -------------------------------------------------------------------------
function out_y = phase_zero_order(in_y, in_phase)
out_y = in_y.*exp(-1i*in_phase);
