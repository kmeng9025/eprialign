% function [out_y, out_pars] = td_get_phase(in_y, in_opt, ...)
% Function optimizes the phase of first dimension of complex array in_y.
%   in_opt.phase_algorithm   - algorithm of phase optimization
%        'manual_zero_order' - rotate on angle in_opt.phase_zero_order
%        'max_real_single'   - indepenent, trace by trace
%        'max_real_all'      - use one phase for all slices
%   ...                      - parameter-value comma separated pairs
%   out_pars                 - structure of supplementary information

% boep, 15/09/2006

function [right_phase_out, out_pars] = ese_getphase(in_y, in_opt, varargin)

if nargin == 2
elseif nargin > 2
  if ~mod(nargin-1,2)
    for kk=1:2:nargin-1
      in_opt=setfield(in_opt, lower(varargin{kk}), varargin{kk+1});
    end
  else error('td_get_phase: Wrong amount of arguments')
  end
else error('Usage: [out_y, out_pars] = td_get_phase(in_y, in_opt, vargin)');
end

out_pars = [];

phase_algorithm = lower(safeget(in_opt, 'phase_algorithm', 'unknown'));

sz    = size(in_y);
sz2   = prod(sz(2:end));

use_echos = safeget(in_opt, 'use_echos', []);
use_echos = use_echos(use_echos <= sz(2));
if isempty(use_echos), use_echos = sz(2); end

% phase is determined independently of method
% for visualization purposes
for ii=1:sz(2)
  for jj=1:sz(3)
    out_pars.phase(ii, jj) = atan2(imag(in_y(in_opt.max_idx,ii, jj)), ...
      real(in_y(in_opt.max_idx,ii, jj)));
  end
end

% out_pars.phase_zero_order_all    phase per trace (array)
% out_pars.phase_zero_order        phase per trace (array or number)

right_phase = zeros(length(use_echos), sz(3));
switch phase_algorithm
  case 'manual_zero_order'
    % manual rotation of the phase
    right_phase = in_opt.phase_zero_order*pi/180*ones(length(use_echos),sz(3));
  case 'max_real_single_smooth'
    % find maximum of the every data smoothed slice
    % change phase to make make maximum at phase 0
    for ii=1:length(use_echos)
      for jj=1:sz(3)
        y_tmp_i = smooth(imag(in_y(:,use_echos(ii), jj)), 5, 'sgolay');
        y_tmp_r = smooth(real(in_y(:,use_echos(ii), jj)), 5, 'sgolay');
        right_phase(ii, jj) = atan2(y_tmp_i(in_opt.max_idx), ...
          y_tmp_r(in_opt.max_idx));
      end
      right_phase(ii, :) = smooth(right_phase(ii, :), 7);
    end
    for jj=1:sz(3)
      right_phase(:, jj) = smooth(right_phase(:, jj), min(3, length(use_echos)));
    end
  case 'max_real_single'
    % find maximum of the every data smoothed slice
    % change phase to make make maximum at phase 0
    for ii=1:length(use_echos)
      for jj=1:sz(3)
        right_phase(ii, jj) = atan2(imag(in_y(in_opt.max_idx,use_echos(ii), jj)), ...
          real(in_y(in_opt.max_idx,use_echos(ii), jj)));
      end
    end
  case 'max_real_all'
    % find maximum of the every data slice
    % calculate everage maximum point and average angle
    % rotate on the calculated angle
    for ii=1:length(use_echos)
      for jj=1:sz(3)
        right_phase(ii, jj) = atan2(imag(in_y(in_opt.max_idx,use_echos(ii), jj)), ...
          real(in_y(in_opt.max_idx,use_echos(ii), jj)));
      end
    end
    right_phase = median(right_phase(:))*ones(length(use_echos),sz(3));
  case 'max_real_poly3'
    % find maximum of the every data smoothed slice
    % change phase to make make maximum at phase 0
    for ii=1:length(use_echos)
      for jj=1:sz(3)
        y_tmp_i = smooth(imag(in_y(:,use_echos(ii), jj)), 5, 'savgol');
        y_tmp_r = smooth(real(in_y(:,use_echos(ii), jj)), 5, 'savgol');
        right_phase(ii, jj) = atan2(y_tmp_i(in_opt.max_idx), ...
          y_tmp_r(in_opt.max_idx));
      end
    end
    for ii=1:length(use_echos)
      p = polyfit(1:sz(3), right_phase(ii, :), 3);
      right_phase(ii, :) = polyval(p, 1:sz(3));
    end    
  case 'max_real_manual'
%     [yyy, out_pars.max_idx_all] = max(abs(yy));
%     out_pars.max_idx     = in_opt.max_idx;
%     for ii=1:sz2
%       val = yy(in_opt.max_idx,ii);
%       phase(ii,1) = atan2(imag(val), real(val));
%       yy(:,ii) = phase_zero_order(yy(:,ii), phase(ii,1));
%     end
%     out_pars.phase_zero_order = phase;
%     out_pars.is_performed = true;
  case 'min_imag_manual'
%     [yyy, out_pars.max_idx_all] = max(abs(yy));
%     out_pars.max_idx     = in_opt.max_idx;
%     % find symmetric interval around echo maximum
%     echo_spread = min(abs(in_opt.max_idx - min(in_opt.echo_area)), abs(in_opt.max_idx - max(in_opt.echo_area)));
%     echo_area   = in_opt.max_idx-echo_spread : in_opt.max_idx+echo_spread;
%     ph = 0;
%     opt = optimset('MaxIter', 500);
%     for ii=1:sz2
%       ph = fminsearch(@optimize_min_imag_manual, ph, opt, yy(echo_area,ii));
%       yy(:,ii) = phase_zero_order(yy(:,ii), ph);
%       [mmm, idx] = max(abs(real(yy(:,ii))));
%       if real(yy(idx,ii)) < 0
%         ph = ph + pi;
%         yy(:,ii) = phase_zero_order(yy(:,ii), pi);
%       end
%       phase(ii,1) = ph;
%     end
%     out_pars.phase_zero_order = phase;
%     out_pars.is_performed = true;
  otherwise
    disp('No phase optimization was performed.');
end

if length(use_echos) ~= sz(2)
  % set average in all rows
  right_phase_out = repmat(sum(right_phase, 1)/numel(use_echos), sz(2), 1);
  right_phase_out(use_echos, :) = right_phase;
else
  right_phase_out = right_phase;
end

% figure(103); clf; hold on
% k = plot(right_phase'*180/pi); set(k, 'LineWidth', 1.5)
% plot(out_pars.phase'*180/pi);
return

% -------------------------------------------------------------------------
function out_y = phase_zero_order(in_y, in_phase)
out_y = in_y.*exp(-1i*in_phase);

function out_res = optimize_min_imag_manual(in_par, in_y)
out_res = abs(sum(imag(phase_zero_order(in_y, in_par(1)))));

%             gau_fun_error = @(x) sqrt(sum((real(yy(:,ii)) - gau_s(xx, x(1), in_opt.max_idx, x(2))).^2));
%             [x] = fminsearch(gau_fun_error, double([mean(real(val)), 50.0]), []);
%             idx = xx(abs(xx - in_opt.max_idx) < 1.2*x(2));
%
%             gau_fun_error1 = @(x) sqrt(sum((real(yy(idx,ii)) - gau_s(xx(idx), x(1), in_opt.max_idx, x(2))).^2));
%             [x] = fminsearch(gau_fun_error1, double([mean(real(val)), 50.0]), []);

%             figure(102); clf; hold on
%             plot(real(yy(:,ii)), 'b'); plot(imag(yy(:,ii)), 'g');
%             XLim = get(gca, 'XLim'); plot(XLim, [0 0], 'k:');
%             plot([1 1]*in_opt.max_idx, [0, max(abs(yy(:,ii)))], 'k:');
%             plot(XLim, min(imag(yy(:,ii)))*[1, 1], 'k:');
%             plot(XLim, max(imag(yy(:,ii)))*[1, 1], 'k:');
%
%             plot(xx(idx),gau_s(xx(idx), x(1), in_opt.max_idx, x(2)), 'r')

%             p = polyfit(xx(idx), imag(yy(idx,ii)), 5);
%             plot(xx(idx), polyval(p, xx(idx)), 'r')
%
%             plot(xx(idx), smooth(imag(yy(idx,ii)), 5, 'savgol'), 'r')
%
%             idx2 = xx(abs(xx - in_opt.max_idx) < 4*x(2));
%             axis([min(idx2), max(idx2), -Inf, Inf])
%
%             pause
