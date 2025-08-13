% function [out_y, out_pars] = kv_phase_optimize(in_y, in_opt, ...)
% Function optimizes the phase of first dimension of complex array in_y.  
%   in_opt.phase_algorithm   - algorithm of phase optimization
%        'manual_zero_order' - rotate on angle in_opt.phase_zero_order
%        'max_real_single'   - indepenent, trace by trace
%        'max_real_all'      - use one phase for all slices 
%   ...                      - parameter-value comma separated pairs
%   out_pars                 - structure of supplementary information

% boep, 15/09/2006

function [out_y, out_pars] = kv_phase_optimize(in_y, in_opt, varargin)

if nargin == 2
elseif nargin > 2
    if ~mod(nargin-1,2)
        for kk=1:2:nargin-1
            in_opt=setfield(in_opt, lower(varargin{kk}), varargin{kk+1});    
        end
    else error('kv_phase_optimize: Wrong amount of arguments')
    end
else error('Usage: [out_y, out_pars] = kv_phase_optimize(in_y, in_opt, vargin)');
end

out_pars = in_opt;

echo_select_algorithm = lower(safeget(in_opt, 'echo_select_algorithm', 'unknown'));
phase_algorithm = lower(safeget(in_opt, 'phase_algorithm', 'unknown'));
fopt=optimset('MaxFunEvals',200);

sz    = size(in_y);
sz2   = prod(sz(2:end));
yy = reshape(in_y, [sz(1), sz2]);
out_pars.is_performed = false;

% find rought estimation of echoe maximum
[yyy, out_pars.max_idx_all] = max(abs(yy));
% out_pars.max_idx_all   maximum per trace (array)
% out_pars.max_idx       returns selected maximum (single number)
% max_idx                for internal use by phase optimization (array)

switch echo_select_algorithm
  case 'manual'
    out_pars.max_idx = in_opt.echo_idx;
    max_idx = in_opt.echo_idx*ones(1, sz2);
    %         use_idx(1:) = in_opt.max_idx;
  case 'statistics'
    % find statistically most reasonable phase and zero point
    if sz2 > 1
      % create approximately phased data
      mean_idx = fix(mean(out_pars.max_idx_all));
      val    = yy(mean_idx, :);
      phase  = atan2(imag(val), real(val));
      yyy = real(phase_zero_order(yy, mean(phase)));
      
      % get approximate width of lorentzian
      lw = double(mean(sum(yyy, 1)./real(val)));
      
      %lorentzian
      lor_z = @(x, amp, x0, lw) amp * lw^2 ./ ((x - x0).^2 + lw^2);
      lor_error = @(x, xx, yy) sqrt(sum((yy - lor_z(xx, x(1), x(2), x(3))).^2));
      
      %gaussian
%       gau_s = @(x, amp, x0, lw) amp/lw/sqrt(2*pi) ./ exp((x - x0).^2/(2*lw^2));
%       gau_error = @(x, xx, yy) sqrt(sum((yy - gau_s(xx, x(1), x(2), x(3))).^2));

      lor = [];
      xxx = double([mean(real(val)), mean_idx, lw]);
      xx  = double(fix(mean_idx-lw/2:mean_idx+lw/2))';
      for ii = 1:sz2
        a = fminsearch(lor_error, xxx, [], xx, double(yyy(xx, ii)));
        lor(end+1) = a(2);
      end
      out_pars.max_idx_all = lor';
      
      % create statistics
      mean_idx = fix(mean(out_pars.max_idx_all));
      std_idx  = std(out_pars.max_idx_all);
      xx = [fix(mean_idx-2*std_idx):.2:fix(mean_idx+2*std_idx)];
      xx = [xx(1)-0.5 xx + 0.5];
      nn = histc(out_pars.max_idx_all(out_pars.max_idx_all > xx(1) & out_pars.max_idx_all < xx(end)),xx);
      [mmm, iidx] = max(nn);
      out_pars.max_idx = floor(xx(iidx)+0.5);
      max_idx = out_pars.max_idx*ones(1, sz2);
    else
      out_pars.max_idx = idx;
    end
  case 'single'
    out_pars.max_idx = idx(1);
    l_idx = size(yy,1);
    for ii=1:length(idx)
      shift = idx(1)-idx(ii);
      yy(:,ii) = [zeros(max(0,shift),1);yy(max(1,1+shift):l_idx+min(shift,0),ii);zeros(max(0,-shift),1);];
    end
  otherwise
    max_idx = out_pars.max_idx_all;
end

% phase is determined independently of method 
% for visualization purposes
val = []; for ii=1:sz2, val(ii,1) = yy(max_idx(ii),ii); end
out_pars.phase_zero_order_all = atan2(imag(val), real(val));

% out_pars.phase_zero_order_all    phase per trace (array)
% out_pars.phase_zero_order        phase per trace (array or number)

switch phase_algorithm
 % manual rotation of the phase
    case 'manual_zero_order'
        yy = phase_zero_order(yy, in_opt.phase_zero_order);
        out_pars.phase_zero_order = in_opt.phase_zero_order(ones(1,sz2),1);
        out_pars.is_performed = true;
 % find maximum of the every data slice
 % change phase to make make maximum at phase 0
    case 'max_real_single'
        out_pars.phase_zero_order = out_pars.phase_zero_order_all;
        for ii=1:sz2
            yy(:,ii) = phase_zero_order(yy(:,ii), out_pars.phase_zero_order(ii,1));
        end
        out_pars.is_performed = true;
% find maximum of the every data slice
% calculate everage maximum point and average angle
% rotate on the calculated angle
    case 'max_real_all'
        for ii=1:sz2
            val = yy(idx(ii),ii);
            phase(ii,1) = atan2(imag(val), real(val));
        end
        out_pars.phase_zero_order = sum(phase)/length(phase);
        yy = phase_zero_order(yy, out_pars.phase_zero_order);
        out_pars.is_performed = true;
    case 'max_real_manual'
        [yyy, out_pars.max_idx_all] = max(abs(yy));
        out_pars.max_idx     = in_opt.max_idx;
        for ii=1:sz2
            val = yy(in_opt.max_idx,ii);
            phase(ii,1) = atan2(imag(val), real(val));
            yy(:,ii) = phase_zero_order(yy(:,ii), phase(ii,1));
        end
        out_pars.phase_zero_order = phase;
        out_pars.is_performed = true;
    case 'min_imag_manual'
        [yyy, out_pars.max_idx_all] = max(abs(yy));
        out_pars.max_idx     = in_opt.max_idx;
        % find symmetric interval around echo maximum
        echo_spread = min(abs(in_opt.max_idx - min(in_opt.echo_area)), abs(in_opt.max_idx - max(in_opt.echo_area)));
        echo_area   = in_opt.max_idx-echo_spread : in_opt.max_idx+echo_spread;
        ph = 0;
        opt = optimset('MaxIter', 500);
        for ii=1:sz2
           ph = fminsearch(@optimize_min_imag_manual, ph, opt, yy(echo_area,ii));
           yy(:,ii) = phase_zero_order(yy(:,ii), ph);
           [mmm, idx1] = max(abs(real(yy(:,ii))));
           if real(yy(idx1,ii)) < 0
               ph = ph + pi;
               yy(:,ii) = phase_zero_order(yy(:,ii), pi);
           end
           phase(ii,1) = ph;
        end
        out_pars.phase_zero_order = phase;
        out_pars.is_performed = true;
    otherwise
        disp('No phase optimization was performed.');
end
out_y = reshape(yy, sz);

% -------------------------------------------------------------------------
function out_y = phase_zero_order(in_y, in_phase)
out_y = in_y.*exp(-i*in_phase);

function out_res = optimize_min_imag_manual(in_par, in_y)
out_res = abs(sum(imag(phase_zero_order(in_y, in_par(1)))));
