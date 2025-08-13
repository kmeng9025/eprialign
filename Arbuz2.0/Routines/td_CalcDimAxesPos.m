% function pst=td_CalcDimAxesPos(row_num, col_num, gaps, sgaps)
% Calculate positions of dimension axes on the figure
% row_num - number of rows
% col_num - number of columns
% gaps    - [horizontal, vertical] gap between plots (in parts of 1)
% sgaps   - [horizontal, vertical] border (in parts of 1)
% ogaps   - [horizontal, vertical] offsets from image
% asize   - [horizontal, vertical] size of axis to make

function [pstx, psty]=td_CalcAxesPos(r,c, gaps, sgaps, ogaps, asize)
% setup axes
pstx=zeros(c,4);
psty=zeros(r,4);

if isempty(gaps), gaps = [0.005 0.005]; end
if isempty(sgaps), sgaps = [0.005 0.005]; end

wdt  = (1.0 - sgaps(1)*2 - (c-1) * gaps(1))/c;
hght = (1.0 - sgaps(2)*2 - (r-1) * gaps(2))/r;
pstx(:, 3) = wdt; pstx(:, 4) = asize(2);
psty(:, 3) = asize(1); psty(:, 4) = hght;

bcorner = 1.0-sgaps(2)-(1:r)*(hght+gaps(2)) + gaps(2);
lcorner = sgaps(1)+(0:c-1)*(wdt+gaps(1));

pstx(:, 1) = lcorner; pstx(:, 2)= sgaps(2) - ogaps(2) - asize(2);
psty(:, 1) = sgaps(1) - ogaps(1) - asize(1); psty(:, 2) = bcorner; 
