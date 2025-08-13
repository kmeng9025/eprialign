% RS_GET_FIELD_PHASE  fit _complex_ EPR signal to Lorenzian
% [cfield, phase, lw] = RS_GET_FIELD_PHASE(x,y,pars)
% x       - Input data [array, 1D] 
% y       - Input data, multiple traces [array, 2D, trace_size x N]
% pars    - Parameters [structure] for future use
% cfield  - Output center field [array, 1D]
% phase   - Output phase [array, 1D, in degree]
% lw      - Output line width [array, 1D]
% See also RS_SFBP, RS_SSCAN_PHASE.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2013-2014
% Contact: epri.uchicago.edu


function [cfield, phase, lw] = rs_get_field_phase(x,y,pars)

sz2 = size(y,2);
cfield = zeros(sz2,1);
phase = zeros(sz2,1);
lw = zeros(sz2,1);

if strcmp(safeget(pars, 'phase_algorithm', 'manual'), 'manual')
  phase = safeget(pars, 'data_phase', 0);
  for ii=1:sz2
    cfield(ii) = 0;
    phase(ii) = phase;
    lw(ii) = 0;
  end
  return;
end

% for ii=1:size(y,2)
%     [~,idx] = max(abs(y(:,1)));
%     cfield(ii) = x(idx);
% end

x = x(:);
bl_range = fix(length(x)/40);
bl = mean(y([1:bl_range, end-bl_range:end], :));

for ii=1:sz2
    yy = y(:,ii) - bl(ii);
    
    x3 = real(sum(yy)*(x(2)-x(1)));
    [maxx, x1] = max(abs(yy));
    lw = x3 / maxx / 2;
    
    f = @(xx) epri_lshape(x, xx(1), [xx(2), xx(2)], 0.94)*xx(3)*exp(-1i*xx(4));
    ferr = @(xx) norm(f(xx)-yy);
    
    xfit = fminsearch(ferr, [x(x1), lw, x3, 0]);
    cfield(ii) = xfit(1);
    phase(ii) = -xfit(4)*180/pi;
    lw(ii) = xfit(2);
%     
%     figure(5); clf;
%     plot(x, real(yy), x, imag(yy)); hold on
%     disp(ferr(xfit))
%     yfit = f(xfit);
%     plot(x, real(yfit), 'r', x, imag(yfit), 'm');
%     interrupt_line = 1;
end