% function pst=epr_CalcAxesPos(row_num, col_num, gaps, sgaps)
% Calculate positions of plots on the figure
% row_num - number of rows
% col_num - number of columns
% gaps    - [horizontal, vertical] gap between plots (in parts of 1)
% sgaps   - [horizontal, vertical] border (in parts of 1)
%         - [horizontal_left, vertical_bottom, horizontal_right, vertical_top]
% use example:
% row_num = col_num = 4; 
% pst=epr_CalcAxesPos(row_num, col_num, [0.1,0.1], [0.1,0.1])
% for ii=1:row_num*col_num, hh(ii)=axes('Position', pst(ii,:)); end

% boep, 2008

function pst=epr_CalcAxesPos(r,c, varargin)

if nargin < 2
  help epr_CalcAxesPos; return;
end

% setup axes
pst=zeros(r*c,4); 

if nargin < 3 || isempty(varargin{1}), 
  gaps = [0.005 0.005]; else gaps = varargin{1}; 
end
if nargin < 4 || isempty(varargin{2}), 
  sgaps = [0.005 0.005]; else sgaps = varargin{2}; 
end

if length(sgaps) == 2
  sgaps = repmat(sgaps, 1, 2);
end

wdt  = (1.0 - sum(sgaps([1,3])) - (c-1) * gaps(1))/c;
hght = (1.0 - sum(sgaps([2,4])) - (r-1) * gaps(2))/r;
pst(:, 3) = wdt; pst(:, 4) = hght;

bcorner = 1.0-sgaps(4)-(1:r)*(hght+gaps(2)) + gaps(2);
lcorner = sgaps(1)+(0:c-1)*(wdt+gaps(1));
pst(:,2) = reshape(bcorner(ones(1,c),:), [r*c,1]);
pst(:,1) = reshape(lcorner(ones(1,r),:)', [r*c,1]);