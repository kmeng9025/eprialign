function res = PxSSPx_decompose(data, peaksPxSSPx, peaksBroken)

simEX = peaksPxSSPx.y;
simBROKEN = peaksBroken.y;

f = @(x)(x(1)*simEX+x(2)*simBROKEN);

clear xx
for ii=1:size(data.y, 2)
  exp_data = data.y(:,ii);
  df = @(x) sum((exp_data-f(x)).^2);
  xx(:,ii) = fminsearch(df, [1,1]);  
%   figure(1); clf
%   plot(data.x, data.y(:,ii)); hold on
%   plot(data.x, f(xx(:,ii)));
%   pause(0.25)
end

res.PxSSPx = xx(1,:);
res.CLEAVED = xx(2,:);
res.time   = data.time;
