function [out] = epri_navigator(out, NAV)

NAVIGATOR_INDEX = 100;

type = safeget(NAV, 'type', []);
n = safeget(NAV, 'n', []);

if strcmp(type, 'all_delay')
  idx = 1:n;
  
  % gradients
  navX = out.G(idx, 1); navX = repmat(navX, [floor(out.nTrace/n+0.5), n]); navX = navX(1:out.nTrace);
  newX = [out.G(:, 1), navX(:)]';
  
  navY = out.G(idx, 2); navY = repmat(navY, [floor(out.nTrace/n+0.5), n]); navY = navY(1:out.nTrace);
  newY = [out.G(:, 2), navY(:)]';

  navZ = out.G(idx, 3); navZ = repmat(navZ, [floor(out.nTrace/n+0.5), n]); navZ = navZ(1:out.nTrace);
  newZ = [out.G(:, 3), navZ(:)]';
  
  out.G = [newX(1:out.nTrace*2-1)' , newY(1:out.nTrace*2-1)' , newZ(1:out.nTrace*2-1)'];
  
  navSweep = out.UnitSweep(idx); navSweep = repmat(navSweep, [floor(out.nTrace/n+0.5), n]); navSweep = navSweep(1:out.nTrace);
  newSweep = [out.UnitSweep(:),navSweep(:)]'; out.UnitSweep = newSweep(1:out.nTrace*2-1)';

  serv = NAVIGATOR_INDEX*ones(out.nTrace, 1);
  newserv = [out.service_idx(:), serv]'; out.service_idx = newserv(1:out.nTrace*2-1)';
  out.nTrace = length(out.service_idx);
end