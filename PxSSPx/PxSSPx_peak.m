function peaks = PxSSPx_peak(x,y,the_state,opts)

peaks.x = x(:);

switch the_state
  case 'cleaved'
    n = length(y);
    idx = [2000:fix(n/2)-2000,fix(n/2)+2000:n-2000]';
    
    xx = x(idx); xx = xx(:);
    yy = double(y(idx)); y = y(:);
    
    A = 22.9;
    OFF = -1.1;
    LW = 1.5;
    LWBB = 10;
    
    F1 = @(x, xx) x(1)*(lshape(xx, -x(2)/2+x(3), x(4), 0, 0.1)+lshape(xx, +x(2)/2+x(3), x(4), 0, 0.1))+x(5)*lshape(xx, 0, x(6), 0, 0);
    F2 = @(x, xx) x(1)*(lshape(xx, -x(2)/2+x(3), x(4), 0, 0.1)+lshape(xx, +x(2)/2+x(3), x(4), 0, 0.1));
    
    opt = optimset('display', 'off');
    x3 = lsqcurvefit(F1,[1, A, OFF, LW, 1, LWBB],xx, yy, [0, A-2, -3, 0.5, 0, 5], [1000, A+2, +3, 4, 10, 50], opt);
    
    peaks.A = x3(2);
    peaks.OFF = x3(3);
    peaks.LW  = x3(4);
    peaks.y = F2(x3(1:4), peaks.x); peaks.y =  peaks.y / sum( peaks.y(:));
    peaks.amp = sum(y(:).*peaks.y) / sum(peaks.y.*peaks.y);
    
    if isfield(opts, 'fig') && ~isempty(opts.fig)
      figure(opts.fig); clf;
      plot(x, y); hold on
      plot(peaks.x, peaks.y*peaks.amp);
    end
  case 'N14'
    n = length(y);
    idx = [2000:fix(n/2)-2000,fix(n/2)+2000:n-2000]';
    
    xx = x(idx); xx = xx(:);
    yy = double(y(idx)); y = y(:);
    
    A1 = -14.0;
    A3 = 15.5;
    OFF = -2.2;
    LW = 1.8;
    LWBB = 10;
    
    idx = (xx > -20 & xx < -11) | (xx > -8 & xx < 3) | (xx > 9 & xx < 18);
    
    % x1,x2,x3 - amplitudes
    F1 = @(x, xx) x(1)*(lshape(xx, -x(2)/2+x(3), x(4), 0, 0.1)+lshape(xx, +x(2)/2+x(3), x(4), 0, 0.1))+x(5)*lshape(xx, 0, x(6), 0, 0);
    F2 = @(x, xx) x(1)*lshape(xx, x(5)+x(4), x(7), 0, 0.2)+...
      x(2)*lshape(xx, x(4), x(7), 0, 0.2)+...
      x(3)*lshape(xx, x(6)+x(4), x(7), 0, 0.2);
    
    
    ymax = max(yy(:))*9;
    opt = optimset('display', 'off');
    x3 = lsqcurvefit(F2,[ymax, ymax, ymax, OFF, A1, A3, LW],xx(idx), yy(idx), ...
      [ymax/2, ymax/2, ymax/2, OFF-3, A1-2, A3-2, 1.8], [ymax*2, ymax*2, ymax*2, OFF+3, A1+2, A3+2, 2.6], opt);
    
    x3, ymax
    
    peaks.A = x3(6)-x3(5);
    peaks.A1 = x3(5);
    peaks.A3 = x3(6);
    peaks.OFF = x3(4);
    peaks.LW  = x3(7);
%     peaks.LW2  = x3(8);
%     peaks.LW3  = x3(9);
    peaks.y = F2(x3, peaks.x); peaks.y =  peaks.y / sum( peaks.y(:));
    peaks.amp = sum(y(:).*peaks.y) / sum(peaks.y.*peaks.y);
    
    if isfield(opts, 'fig') && ~isempty(opts.fig)
      figure(opts.fig); clf;
      plot(x, y); hold on
      plot(peaks.x, peaks.y*peaks.amp);
      xlabel('Field [G]');
    end
  otherwise
    LW = safeget(opts, 'LW', 7);
    LWex = 17;
    OFF = safeget(opts, 'CF', 0);
    A   = safeget(opts, 'A', 20);
    
    peaks.y = (lshape(x, -A/2+OFF, LW, 0, 0.4)+lshape(x, +A/2+OFF, LW, 0, 0.4))+...
      LWex/LW*1.7*lshape(x, OFF, LWex, 0, 0.4) + ...
      0;
    peaks.y = peaks.y(:) / sum(peaks.y(:));
    peaks.amp = sum(y(:).*peaks.y) / sum(peaks.y.*peaks.y);
    
    if isfield(opts, 'fig') && ~isempty(opts.fig)
      figure(opts.fig); clf;
      plot(x, y); hold on
      plot(peaks.x, peaks.y*peaks.amp);
    end
end