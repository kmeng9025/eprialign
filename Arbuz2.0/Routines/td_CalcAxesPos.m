% function pst=td_CalcAxesPos(row_num, col_num, gaps, sgaps)
% Calculate positions of plots on the figure
% row_num - number of rows
% col_num - number of columns
% gaps    - [horizontal, vertical] gap between plots (in parts of 1)
% sgaps   - [horizontal, vertical] border (in parts of 1)

function pst=td_CalcAxesPos(r,c,gaps, sgaps)
% setup axes
pst=zeros(r*c,4);  %6 subplots

if isempty(gaps), gaps = [0.005 0.005]; end
if isempty(sgaps), sgaps = [0.005 0.005]; end

wdt  = (1.0 - sgaps(1)*2 - (c-1) * gaps(1))/c;
hght = (1.0 - sgaps(2)*2 - (r-1) * gaps(2))/r;
pst(:, 3) = wdt; pst(:, 4) = hght;

bcorner = 1.0-sgaps(2)-(1:r)*(hght+gaps(2)) + gaps(2);
lcorner = sgaps(1)+(0:c-1)*(wdt+gaps(1));
pst(:,2) = reshape(bcorner(ones(1,c),:), [r*c,1]);
pst(:,1) = reshape(lcorner(ones(1,r),:)', [r*c,1]);