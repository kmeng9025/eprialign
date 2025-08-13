function fit = PxSSPX_fit_imaging(Input_var);



Time_course_data = Input_var(:);
time1 =  ((1:size(Time_course_data)).*31)';
dt = mean(diff(time1));



F1 = @(x, T1) x(1)*(1-exp(-x(2).*(T1))).*exp(-x(3).*(T1))+x(4);

T1 = time1;

kin1 = [];
kin2 = [];

opt = optimset('display', 'off');
lb = [0, 1/1000, 1/20000, 0];
ub = [1000, 1/50, 1/1000, 60];

 [x1,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA,JACOBIAN] = lsqcurvefit(F1,[0.03, 1/100, 1/2000, 0],T1, double(Time_course_data), [], [], opt);  
  
  kin1 = 1/x1(2);
  kin2 = 1/x1(3);
  
  degree_of_freedom = length(T1)-1;
  Sigma = RESNORM*inv(JACOBIAN'*JACOBIAN )/ degree_of_freedom;
  se = sqrt(diag(Sigma));
  
%   if kin1 < 50
%       kin1 = kin2
%       kin2= 10000
%   end

  
  figure(103); clf;
  plot(T1, Time_course_data, 'ro', T1, F1(x1, T1), 'b'); hold on
  conf = nlparci(x1,RESIDUAL,'jacobian',JACOBIAN)
  plot(T1, F1(conf(:,1), T1), 'b:', T1, F1(conf(:,2), T1), 'b:'); hold on
  
% % %   text(0.5, 0.62, sprintf('%d %5.1fs / %5.1fs', ii, 1/x1(2), 1/x1(3)), 'units', 'normalized');
% %   pause(0.25)
fit.kin1 = kin1;
fit.kin2 = kin2;
fit.Error_kin1 = se(2)
fit.Error_kin2 = se(3)

% hold
% plot(time1,Time_course_data)
% plot(time1,(F1(x1,time1)))
% 
% pause(1.5)
% clf
end

% fit_dx1 = @(x, T1) (1-exp(-x(2).*(T1))).*exp(-x(3).*(T1));
% fit_dx2 = @(x, T1) x(1)*exp(-x(3).*(T1)).*(-exp(-x(2).*(T1))).*(-T1);
% fit_dx3 = @(x, T1) x(1)*(1-exp(-x(2).*(T1))).*exp(-x(3).*(T1)).*(-T1);
% fit_dx4 = @(x, T1) ones(size(T1));

%     residual_std = fval/sqrt(degree_of_freedom);
%     J = [fit_exp_dx1(x); fit_exp_dx2(x)]';
%     Sigma = residual_std^2*inv(J'*J);
%     se = sqrt(diag(Sigma))';













% F1 = @(x, T1) x(1)*exp(-x(2).*T1) + x(3)*exp(-x(4).*T1);
% F2 = @(x, T1) x(1)*exp(-x(2).*T1)+x(3);
% F3 = @(x, T1) x(1)*(1-exp(-x(2).*T1)).*exp(-x(3).*T1)+x(4);
% 
% switch the_mode
%   case 'all'
%     opt = optimset('display', 'off');
%     fp = safeget(opts, 'fp', 1);
%     
%     the_time = res.time(fp:end);
%     the_time = the_time(:);
%     CLEAVED = res.CLEAVED(fp:end);
%     CLEAVED = CLEAVED(:);
%     PxSSPx = res.PxSSPx(fp:end);
%     PxSSPx = PxSSPx(:);
% 
%     x1 = lsqcurvefit(F3,[1000, 1/250, 1/15000, 0],the_time,CLEAVED, [0 1/1300, 1/10000, 0], [1000, 1/130, 1/400, 100], opt);
%     fit.k_F3 = 1/x1(2);
%     fit.c_F3 = 1/x1(3);
%     
%     x2 = lsqcurvefit(F2,[100, 1/300, 0],the_time,PxSSPx, [0 1/2500, 0], [1000, 1/130, 100], opt);
%     fit.k_F2 = 1/x2(2);
%     
%     
%     if isfield(opts, 'fig') && ~isempty(opts.fig)
%       figure(opts.fig); clf;
%       plot(res.time, res.CLEAVED, 'o', res.time, F3(x1, res.time), '-'); hold on
%       plot(res.time, res.PxSSPx, 'o', res.time, F2(x2, res.time), '-')
%       text(0.5, 0.77, sprintf('From PxSH buildup: %5.1fs', 1/x1(2)), 'units', 'normalized', 'FontSize', 14);
%       text(0.5, 0.71, sprintf('From PxSSPx decay: %5.1fs', 1/x2(2)), 'units', 'normalized', 'FontSize', 14);
%       text(0.5, 0.65, sprintf('Clearence: %5.1fs', 1/x1(3)), 'units', 'normalized', 'FontSize', 14);
%       xlabel('Time [s]');
%       ylabel('Signal [a.u.]');
%       axis('tight');
%       legend({'Data', 'Fit', 'Data', 'Fit'});
%       if isfield(opts, 'title'), title(opts.title, 'interpreter', 'none'); end
%     end
%   case ''
% end
% 
