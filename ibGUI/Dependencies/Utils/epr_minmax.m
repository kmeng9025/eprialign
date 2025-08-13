% res = epr_minmax(data <, method_str>)
% Get minimum/maximum of data
%   data - any data
%   method_str - 'all' for [min(data(:)), max(data(:))]
%              - 'n%'  for boundaries that exclude 3% of data highest and
%              lowest values.
% Example: epr_minmax(1:100, '3%') -> [4,96]

% boep, 08

function varargout = epr_minmax(data, method_str)

if nargin == 0
  help epr_minmax; return;
elseif nargin == 1
  method = 'all';
elseif ~isempty(findstr(method_str, '%'))
  method = '%';
  method_par = sscanf(method_str, '%f%%') * 0.01;
end

data = data(:);
data_min = min(data);
data_max = max(data);
if data_min == data_max, method = 'eq'; end 

switch method
  case '%',
    iter = 0;
    while iter < 5
      data_edges = linspace(data_min, data_max, 500);
      data_hist = hist(data, data_edges);
      data_hist = data_hist / sum(data_hist);
%       figure; bar(data_edges, data_hist);
      [mmax, idx] = max(data_hist);
      
      % suppressing the highly repeating point
      if mmax > 0.9,
        data_hist(idx) = 0.001;
        data_hist = data_hist * 0.899;
      end
      
%       kk = find(data_hist > 0.0001);
%       data_min = data_edges(max(1,min(kk)-1));
%       data_max = data_edges(max(kk));
      
      data_hist_int = cumsum(data_hist);
      data_hist_int = data_hist_int / data_hist_int(end);
      iter = iter + 1;
    end
    idx = find(data_hist_int <  1 - method_par & data_hist_int > method_par);
    if isempty(idx)
      res = [0 1];
    else
      res = [data_edges(idx(1)), data_edges(idx(end))];
    end
  case 'eq', res = [0, 1];
  otherwise, res = [data_min, data_max];
end

if nargout == 2
  varargout{1} = res(1); varargout{2} = res(2);
else
  varargout{1} = res;
end