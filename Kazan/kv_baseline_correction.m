% function [out_y, out_pars] = kv_baseline_correction(in_y, in_opt, ...)
% Function searched the baseline and brings it's average to zero
%   in_opt.baseline_algorithm  - algorithm of baseline correction
%          'zero_single'     - indepenent, trace by trace
%          'zero_all'        - use one phase for all slices 
%   in_opt.baseline_area_algorithm - algorithm of baseline area search
%          'manual'          - defined manually
%   in_opt.baseline_area     - baseline area
%   ...                      - parameter-value comma separated pares
%   out_pars                 - structure of supplementary information

function [out_y, out_pars] = kv_baseline_correction(in_y, in_opt, varargin)

if nargin == 2
elseif nargin > 2
    if ~mod(nargin-1,2)
        for kk=1:2:nargin-1
            in_opt=setfield(in_opt, lower(varargin{kk}), varargin{kk+1});    
        end
    else error('kv_baseline_correction: Wrong amount of arguments')
    end
else error('Usage: [out_y, out_pars] = kv_baseline_correction(in_y, in_opt, vargin)');
end

out_pars = in_opt;
out_pars.is_performed = 1;

baseline_algorithm = lower(safeget(in_opt, 'baseline_algorithm', 'unknown'));
baseline_area_algorithm = lower(safeget(in_opt, 'baseline_area_algorithm', 'unknown'));
baseline_area = lower(safeget(in_opt, 'baseline_area', 'unknown'));
  
fopt=optimset('MaxFunEvals',200);

sz    = size(in_y);
if length(sz) == 2, sz = [sz(1),1,sz(2),1]; 
elseif length(sz) == 3, sz = [sz,1];
end
sz2   = prod(sz(2:end));
yy = reshape(in_y, [sz(1), sz2]);

if numel(baseline_area) == 1
  baseline_area = sz(1)-baseline_area:sz(1);
  out_pars.baseline_area = baseline_area;
end

switch baseline_area_algorithm
    case 'auto'
end
baseline_area_length = length(baseline_area);

if any(baseline_area > sz(1))
  error('Baseline area exceed last data point.');
end

switch(baseline_algorithm)
    case 'zero_single'
        for ii=1:sz2
            P1 = polyfit(baseline_area',real(yy(baseline_area,ii)),0);
            P2 = polyfit(baseline_area',imag(yy(baseline_area,ii)),0);
            bl = P1 + 1i*P2;
            yy(:,ii) = yy(:,ii) - bl;
            out_pars.baseline_zero(ii,1) = bl;
        end
    case 'zero_all'
        whole_trace = baseline_area_length*sz2;
        yyy = reshape(yy(baseline_area,:), [whole_trace, 1]);
        P1 = polyfit((1:whole_trace)',real(yyy),0);
        P2 = polyfit((1:whole_trace)',imag(yyy),0);
        out_pars.baseline_zero = P1 + 1i*P2;
        yy = yy - out_pars.baseline_zero;
    case 'first_single'
        for ii=1:sz2
            P1 = polyfit(baseline_area',real(yy(baseline_area,ii)),1);
            P2 = polyfit(baseline_area',imag(yy(baseline_area,ii)),1);
            bl = polyval(P1,(1:sz(1))') + 1i*polyval(P2,(1:sz(1))');
            yy(:,ii) = yy(:,ii) - bl;
            out_pars.baseline_zero(ii,1) = 0;
        end
    otherwise
        out_pars.baseline_zero = 0;
        out_pars.is_performed = 0;
        disp('No baseline correction was performed.');
end
out_y = reshape(yy, sz);
if out_pars.is_performed
  out_pars.baseline_zero = reshape(out_pars.baseline_zero, [sz(3), sz(2), sz(4)]);
else
  out_pars.baseline_zero = [];
end

