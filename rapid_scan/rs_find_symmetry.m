% --------------------------------------------------------------------
function shift = rs_find_symmetry(in_y, c_shift)

nPts = size(in_y, 1);
nTr = size(in_y, 2) / 2;
in_y = reshape(in_y, [nPts, 2, nTr]);
[mhalf1, mhalf2] = rs_split_trace(in_y(:,1,:), c_shift);
[phalf1, phalf2] = rs_split_trace(in_y(:,2,:), c_shift);

shifts = zeros(nTr,2);
for ii=1:nTr
  shifts(ii,1) = find_symmetry(mhalf1(:,ii),phalf1(:,ii));
  shifts(ii,2) = find_symmetry(mhalf2(:,ii),phalf2(:,ii));
end
shift = fix(mean(shifts(:)));

function res = find_symmetry(trace1, trace2)
Npt = length(trace1);

% Assume that line should be very close to the center
% Take into account there is more signal in dispersion
dcs1 = abs(trace1 - circshift(trace1, 4)); 
dcs1(fix([1:Npt/5,Npt*4/5:Npt])) = 0;
dcs2 = abs(trace2 - circshift(trace2, 4)); 
dcs2(fix([1:Npt/5,Npt*4/5:Npt])) = 0;

[a, zf_idx(1)] = max(dcs1);
[a, zf_idx(2)] = max(dcs2);

the_shift = fix(diff(zf_idx)/2); 
shifts = -the_shift + (-30:30);

%use only real part as it is sharper
err = [];
for ii= shifts
  compare_idx = zf_idx(1) + ii + Npt/16*(-1:1);
  tr2a = circshift(trace1, -ii);
  tr2b = circshift(trace2, ii);
  err(end+1) = sum((real(tr2a(compare_idx) - tr2b(compare_idx))).^2);
end
[mm, idx] = min(err);
res = shifts(idx);

% figure(200); clf; hold on
% iplot(1:Npt, circshift(trace1, -res), 1)
% iplot(1:Npt, circshift(trace2, res), 3)
% axis tight
