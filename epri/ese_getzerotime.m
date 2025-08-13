% [max_idx, out_pars] = td_get_zero_time(yy, in_pars)
% max_idx                returns prefered maximum 
% out_pars               parameters for report generation
%   -  max_idx_all       maximum per trace (array)

function [max_idx, out_pars] = ese_getzerotime(yy, in_pars)

sz = size(yy);
% Size of y
%   sz(1) - time domain
%   sz(2) - signals
%   sz(3) - projections

echo_select_algorithm = lower(safeget(in_pars, 'echo_select_algorithm', 'unknown'));

use_echos = safeget(in_pars, 'use_echos', []);
use_echos = use_echos(use_echos <= sz(2));
if isempty(use_echos), use_echos = sz(2); end

switch echo_select_algorithm
  case 'manual'
    max_idx = in_pars.echo_idx;

    % find rought estimation of echoe maximum
    [yyy, out_pars.max_idx_all] = max(abs(yy));
    out_pars.max_idx_all = out_pars.max_idx_all(:);
  case 'statistics'
    yy = yy(:,use_echos,:);
    sz2 = length(use_echos)*sz(3);
    yy = reshape(yy, [sz(1), sz2]);

    % find rought estimation of echoe maximum
    [yyy, out_pars.max_idx_all] = max(abs(yy));
    
    % clean up outliers
    % clean up very distant outliers
    zero_std = std(out_pars.max_idx_all);
    zero_median = median(out_pars.max_idx_all);
    idx_filtered_zero = (out_pars.max_idx_all >= (zero_median - zero_std)) & ...
      (out_pars.max_idx_all <= (zero_median + zero_std));
    
    % rough clean up
    zero_std = std(out_pars.max_idx_all(idx_filtered_zero));
    zero_median = median(out_pars.max_idx_all(idx_filtered_zero));
    idx_filtered_zero = (out_pars.max_idx_all >= (zero_median - 2*zero_std)) & ...
      (out_pars.max_idx_all <= (zero_median + 2*zero_std));

    % fine clean up
    zero_std = std(out_pars.max_idx_all(idx_filtered_zero));
    zero_median = median(out_pars.max_idx_all(idx_filtered_zero));
    idx_filtered_zero = (out_pars.max_idx_all >= (zero_median - zero_std)) & ...
      (out_pars.max_idx_all <= (zero_median + zero_std));
%     max_idx = median(out_pars.max_idx_all(idx_filtered_zero));
    max_idx = floor(mean(out_pars.max_idx_all(idx_filtered_zero)) + 0.5);
  case 'single'
    out_pars.max_idx = idx(1);
    l_idx = size(yy,1);
    for ii=1:length(idx)
      shift = idx(1)-idx(ii);
      yy(:,ii) = [zeros(max(0,shift),1);yy(max(1,1+shift):l_idx+min(shift,0),ii);zeros(max(0,-shift),1);];
    end
  otherwise
    % find rought estimation of echoe maximum
    [yyy, out_pars.max_idx_all] = max(abs(yy));

    max_idx = mean(out_pars.max_idx_all);
end

out_pars.max_idx = max_idx;

% -------------------------------------------------------------------------
function out_y = phase_zero_order(in_y, in_phase)
out_y = in_y.*exp(-1i*in_phase);