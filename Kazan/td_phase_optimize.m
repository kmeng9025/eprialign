% function [out_y, out_pars] = kv_phase_optimize(in_y, in_opt, ...)
% Function optimizes the phase of first dimension of complex array in_y.  
%   in_opt.phase_algorithm   - algorithm of phase optimization
%        'manual_zero_order' - rotate on angle in_opt.phase_zero_order
%        'max_real_single'   - indepenent, trace by trace
%        'max_real_all'      - use one phase for all slices 
%   ...                      - parameter-value comma separated pairs
%   out_pars                 - structure of supplementary information

% boep, 15/09/2006

function [out_y, out_pars] = td_phase_optimize(in_y, in_opt, vargin)

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

gau_s = @(x, amp, x0, lw) amp/lw/sqrt(2*pi) ./ exp((x - x0).^2/(2*lw^2));

phase_algorithm = lower(safeget(in_opt, 'phase_algorithm', 'unknown'));
fopt=optimset('MaxFunEvals',200);

sz    = size(in_y);
sz2   = prod(sz(2:end));
yy = reshape(in_y, [sz(1), sz2]);
out_pars.is_performed = false;

phase_average_echo = safeget(in_opt, 'phase_average_echo', false);
use_echos = safeget(in_opt, 'use_echos', 1:sz(2));

% phase is determined independently of method 
% for visualization purposes
val = []; for ii=1:sz2, val(ii,1) = yy(in_opt.max_idx,ii); end
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
%         xx = (1:sz(1))';
        for ii=1:sz2

%             gau_fun_error = @(x) sqrt(sum((real(yy(:,ii)) - gau_s(xx, x(1), in_opt.max_idx, x(2))).^2));
%             [x] = fminsearch(gau_fun_error, double([mean(real(val)), 50.0]), []);
%             idx = xx(abs(xx - in_opt.max_idx) < 1.2*x(2));
% 
%             gau_fun_error1 = @(x) sqrt(sum((real(yy(idx,ii)) - gau_s(xx(idx), x(1), in_opt.max_idx, x(2))).^2));
%             [x] = fminsearch(gau_fun_error1, double([mean(real(val)), 50.0]), []);
            
            y_tmp_i = smooth(imag(yy(:,ii)), 5, 'savgol');
            y_tmp_r = smooth(real(yy(:,ii)), 5, 'savgol');
            
            out_pars.phase_zero_order(ii, 1) = atan2(y_tmp_i(in_opt.max_idx), ...
              y_tmp_r(in_opt.max_idx));
        end
        
        if phase_average_echo
        end
        
        for ii=1:sz2
            yy(:,ii) = phase_zero_order(yy(:,ii), out_pars.phase_zero_order(ii,1));

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
           [mmm, idx] = max(abs(real(yy(:,ii))));
           if real(yy(idx,ii)) < 0
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
figure(103); clf
plot(reshape(out_pars.phase_zero_order_all*180/pi, [sz(2), sz(3)])')
out_y = reshape(yy, sz);

% -------------------------------------------------------------------------
function out_y = phase_zero_order(in_y, in_phase)
out_y = in_y.*exp(-i*in_phase);

function out_res = optimize_min_imag_manual(in_par, in_y)
out_res = abs(sum(imag(phase_zero_order(in_y, in_par(1)))));
