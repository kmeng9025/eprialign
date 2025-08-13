% ddd = export.Images{1}.Image;
% save('d:\ProcessedData\2015\150420\kinetics2.mat', 'ddd')

data = load('d:\ProcessedData\2015\150420\time_course.mat');
% data = load('d:\ProcessedData\2015\150420\time_course2.mat');
data = data.ddd;
fit_mask = load('d:\ProcessedData\2015\150420\fit_mask.mat');
the_mask = fit_mask.Mask;

fit_data = data(repmat(the_mask, 1,1,1,75));
fit_data = reshape(fit_data, [length(fit_data)/75, 75])';

%%
dt = mean(diff(time1));

idx = 3:size(fit_data, 1);
ndata = size(fit_data, 2);
F1 = @(x, T1) x(1)*(1-exp(-x(2).*(T1))).*exp(-x(3).*(T1))+x(4);

T1 = dt * idx';

kin1 = [];
kin2 = [];

opt = optimset('display', 'off');
lb = [0, 1/1000, 1/20000, 0];
ub = [1000, 1/50, 1/1000, 60];
for ii=1:ndata
   the_trace = fit_data(idx,ii);
  x1 = lsqcurvefit(F1,[0.03, 1/100, 1/2000, 0],T1,double(the_trace), lb, ub, opt);  
  kin1(ii) = 1/x1(2);
  kin2(ii) = 1/x1(3);
  disp(ii);
  
%   figure(103); clf;
%   plot(T1, the_trace, 'ro', T1, F1(x1, T1), 'b'); hold on
%   text(0.5, 0.62, sprintf('%d %5.1fs / %5.1fs', ii, 1/x1(2), 1/x1(3)), 'units', 'normalized');
%   pause(0.25)
end

%%

kin_image = zeros(64,64,64);
kin_image(the_mask) = kin2;
ibGUI(struct('kCleavage', kin_image, 'Mask', the_mask))
