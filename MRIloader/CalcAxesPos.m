function pst=CalcAxesPos(r,c,gaps, sgaps)
% pst=CalcAxesPos(row_num, col_num, gaps, sgaps) 
% Calculate positions of subplots on the figure, without using the subplots
% "feature" of Matlab
% r - number of rows
% c - number of columns
% gaps    - [horizontal, vertical] gap between subplots
% default is 0.005
% sgaps   - [horizontal, vertical] border around all plots
% default is 0.006 and 0.009
% assumes that the units of position for the axes is normalized (the
% default)
% use output pst with axes, e.g.,  axes('Position',pst(i,:)), imagesc(...
% for the i-th subplot.
% C. Haney modified code from Matlab newsgroup December 2005
% based on post from Derek Goring & Omer Demirkaya Dec 17 2002
% Modified to use side gaps, B. Epel June 2006


% setup axes
pst=zeros(r*c,4);  %r x c subplots with [left bottom width height]


if isempty(gaps), gaps = [0.005 0.005]; end
if isempty(sgaps), sgaps = [0.006 0.009]; end

wdt  = (1.0 - sgaps(1)*2 - (c-1) * gaps(1))/c;
hght = (1.0 - sgaps(2)*2 - (r-1) * gaps(2))/r;
pst(:, 3) = wdt; pst(:, 4) = hght;

bcorner = 1.0-sgaps(2)-(1:r)*(hght+gaps(2)) + gaps(2); lcorner = sgaps(1)+(0:c-1)*(wdt+gaps(1));
pst(:,2) = reshape(bcorner(ones(1,c),:), [r*c,1]);
pst(:,1) = reshape(lcorner(ones(1,r),:)', [r*c,1]);