function fit = PxSSPX_fit_imaging(res, the_mode, opts)

F1 = @(x, T1) x(1)*exp(-x(2).*T1) + x(3)*exp(-x(4).*T1);
F2 = @(x, T1) x(1)*exp(-x(2).*T1) +x(3);
F3 = @(x, T1) x(1)*(1-exp(-x(2).*T1)).*exp(-x(3).*T1)+x(4);

switch the_mode
  case 'all'
    opt = optimset('display', 'off');
    fp = safeget(opts, 'fp', 1);
    
    the_time = res.time(fp:end);
    the_time = the_time(:);
    CLEAVED = res.CLEAVED(fp:end);
    CLEAVED = CLEAVED(:);
    PxSSPx = res.PxSSPx(fp:end);
    PxSSPx = PxSSPx(:);

    % cleaved kitetix (2 exp)
    [x1,RESNORM,RESIDUAL_1,EXITFLAG,OUTPUT,LAMBDA,JACOBIAN]  = lsqcurvefit(F3,[1000, 1/150, 1/15000, 0],the_time,CLEAVED, [0 1/1300, 1/15000, 0], [1000, 1/100, 1/300, 0.1], opt);
    fit.k_PxSH = x1(2)*1E3;
    fit.k_clr = x1(3)*1E3;
    fit.a1 = x1(1);
          
     degree_of_freedom = length(the_time)-1;
     Sigma = RESNORM*inv(JACOBIAN'*JACOBIAN )/ degree_of_freedom;
     se = sqrt(diag(Sigma));
     fit.error_of_k_PxSH = se(2) * 1E3 + 0;
     fit.Error_in_kclr = se(3) * 1E3 + 0;
     fit.Error_a1 = se(1) + 0;
  
    
     % cleavage (1 kinetix)
    [x2,RESNORM,RESIDUAL_2,EXITFLAG,OUTPUT,LAMBDA,JACOBIAN] = lsqcurvefit(F2,[100, 1/300, 0],the_time,PxSSPx, [0 1/2500, 0], [1000, 1/100, 100], opt);
    fit.k_PxSSPx_cleavage = x2(2)*1E3;
    fit.a2 = x2(1);
    fit.a1a2 = x1(1)/x2(1);
    
     degree_of_freedom = length(the_time)-1;
     Sigma = RESNORM*inv(JACOBIAN'*JACOBIAN )/ degree_of_freedom;
     se = sqrt(diag(Sigma));
     fit.Error_in_Cleavage = se(2) * 1E3 + 0;
     fit.Error_a2 = se(1) + 0;
     
    fit.Error_a1a2 =  fit.a1a2 * sqrt((fit.Error_a1/fit.a1).^2 + (fit.Error_a2/fit.a2).^2);
    
    fit.F3 = F3(x1, res.time);
    fit.F2 = F2(x2, res.time);
    if isfield(opts, 'fig') && ~isempty(opts.fig)
      figure(opts.fig); clf;
      plot(res.time, res.CLEAVED, 'o', res.time, F3(x1, res.time), '-'); hold on
      plot(res.time, res.PxSSPx, 'o', res.time, F2(x2, res.time), '-')
      text(0.5, 0.77, sprintf('From PxSH buildup: %5.1fs', 1/x1(2)), 'units', 'normalized', 'FontSize', 14);
      text(0.5, 0.71, sprintf('From PxSSPx decay: %5.1fs', 1/x2(2)), 'units', 'normalized', 'FontSize', 14);
      text(0.5, 0.65, sprintf('Clearence: %5.1fs', 1/x1(3)), 'units', 'normalized', 'FontSize', 14);
      xlabel('Time [s]');
      ylabel('Signal [a.u.]');
      axis('tight');
      legend({'Data', 'Fit', 'Data', 'Fit'});
      if isfield(opts, 'title'), title(opts.title, 'interpreter', 'none'); end
    end
  case 'buildup'
    opt = optimset('display', 'off');
    fp = safeget(opts, 'fp', 1);
    
    the_time = res.time(fp:end);
    the_time = the_time(:);
    CLEAVED = res.CLEAVED(fp:end);
    CLEAVED = CLEAVED(:);

    [x1,resnorm,RESIDUAL] = lsqcurvefit(F3,[1000, 1/150, 1/15000, 0],the_time,CLEAVED, [0 1/1300, 1/10000, 0], [1000, 1/100, 1/300, 0.1], opt);
    
    fit.k_F3 = x1(2);
    fit.c_F3 = x1(3);
end

