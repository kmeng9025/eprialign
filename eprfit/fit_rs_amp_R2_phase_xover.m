% [out_pars, out_errs] = fit_cw_amp_R2_phase_xover(B, yy, mat_fit_info)
% 4 parameters fit of spectral data
% B :  magnetic field
% yy:  spectra in 2D array, first dimension is spectral
% mat_fit_info.spin_probe: spin probe
% mat_fit_info.ModFrequency: modulation frequency
% mat_fit_info.ModAmplitude: modualtion amplitude

function [out_pars, out_errs, out_fit] = fit_rs_amp_R2_phase_xover(B, yy, mat_fit_info)

sz = size(yy);
spin_probe = safeget(mat_fit_info, 'spin_probe', 'OX063H');
h = -1;

if strcmpi(spin_probe, 'LORENTZIAN')
  switch safeget(mat_fit_info, 'fit_method', 'default')
    case 'lookup_general'
      pars.func = @(x, x0, fwhm, ph) real(exp(-1i*ph)*epr_Lorentzian(x, x0, fwhm));
      pars.par1 = mat_fit_info.fit_x0_min_max; % line offset
      pars.par2 = mat_fit_info.fit_LW_min_max;
      pars.par3 = mat_fit_info.fit_ph_min_max; % phase
      pars.x  =  B(:);

      tic
      [~, pars] = fit_lookup3([], pars, 0);
      res = fit_lookup3(yy, pars, 1);
      toc

      out_pars = zeros(4,sz(2));
      out_errs = zeros(4+1,sz(2));

      out_pars(1,:) = res(:,1);
      out_pars(2,:) = res(:,2);
      out_pars(3,:) = res(:,3);
      out_pars(4,:) = res(:,4);

      out_fit = [];
    otherwise
      out_pars = zeros(4,sz(2));
      out_errs = zeros(4+1,sz(2));
      
      F = @(x, B) x(1)*real(exp(-1i*x(4))*epr_Lorentzian(B, x(2), x(3)));
      
      xx = [1, mean(mat_fit_info.fit_x0_min_max), mean(mat_fit_info.fit_LW_min_max), mean(mat_fit_info.fit_ph_min_max)];
      opt = optimoptions('lsqcurvefit', 'Display', 'off');
      for ii=1:size(yy,2)
        
        [res,resnorm,~,exitflag, output] = lsqcurvefit(F,xx,B(:),double(yy(:,ii)),[],[],opt);
        out_pars(1,ii) = res(1);
        out_pars(2,ii) = res(2);
        out_pars(3,ii) = res(3);
        out_pars(4,ii) = res(4);
        
%         figure(100); clf;
%         plot(B(:), yy(:,ii), 'o'); hold on;
%         plot(B(:), F(res, B(:)));
%         text(0.1, 0.9, sprintf('%i XOVER=%4.2f LW=%4.2f PH=%4.2f', ii, res(2), res(3), res(4)), 'units', 'normalized')
%         pause(0.5)
      end
      
  end
elseif strcmpi(spin_probe, 'VOIGTIAN')
  sigma = mat_fit_info.fit_gauss_LW; % [G]
  pars.func = @(x, x0, fwhm, ph) real(exp(-1i*ph/180*pi)*epr_Voigtian(x, x0, fwhm, sigma));
  pars.par1 = mat_fit_info.fit_x0_min_max; % line offset
  pars.par2 = mat_fit_info.fit_LW_min_max;
  pars.par3 = mat_fit_info.fit_ph_min_max;       % phase
  pars.x  =  B(:);
  
  tic
  [~, pars] = fit_lookup_3par([], pars, 0);
  res = fit_lookup_3par(yy, pars, 1);
  toc
  
  out_pars = zeros(4,sz(2));
  out_errs = zeros(4+1,sz(2));
  
  out_pars(1,:) = res(:,1);
  out_pars(2,:) = res(:,2);
  out_pars(3,:) = res(:,3);
  out_pars(4,:) = res(:,4);
  
  out_fit = [];
else
  % Select model
  [shf_model, shf_pars] = cw_shf_model(spin_probe);
  [pattern, resolution] = cw_shf(shf_pars);
  Bshf = (1:length(pattern))' * resolution; Bshf = Bshf - mean(Bshf);

  warning off;
  try
    fpars.B         = B;
    fpars.Bshf      = Bshf;
    fpars.pattern   = pattern;
    fpars.idx = [1,2,3]';
    B0              = mean(B);

    out_pars = zeros(4,sz(2));
    out_errs = zeros(4+1,sz(2));
    out_pars(3,:) = 0.01;

    if sz(2) > 100
      % Estimation
      p2p = max(yy) - min(yy);
      idx = p2p > median(p2p) & p2p < median(p2p)+0.25*std(p2p);
      estimation_idx = find(idx);
      estimation_n = length(estimation_idx);
      % reduce estimation points to make them below 300
      while estimation_n > 300
        idx(estimation_idx(1:2:end))=0;
        estimation_idx = find(idx);
        estimation_n = length(estimation_idx);
      end
      idx = not(idx);
      other_idx = find(idx);
      other_n = length(other_idx);

      h = waitbar(0,'Please wait...','Name',sprintf('Estimation (%i spectra) is in progress', estimation_n));

      k = 0;
      for ii=estimation_idx
        fpars.yy = real(double(yy(:,ii)));
        fmin = @(x) sqrt(sum((fpars.yy - simple_fit(x,fpars)).^2));
        out_pars(1:4,ii) = fminsearch(fmin, [max(abs(fpars.yy)), B0, 20e-3, 0]);
        out_fit = simple_fit(out_pars(1:4,ii),fpars);
        %       out_errs(2:4,ii)=xx_err;
        %       out_errs(5,ii)=sqrt(sum((fpars.yy/out_pars(1,ii)-out_fit).^2))/sum(abs(out_fit));
        waitbar(k/estimation_n, h); k = k+1;
      end
      if ishandle(h), delete(h); end

      out_pars(2,:) = median(out_pars(2,estimation_idx));
      out_pars(3,:) = median(out_pars(3,estimation_idx));
      out_pars(4,:) = median(out_pars(4,estimation_idx));

      % Fitting
      h = waitbar(0,'Please wait...','Name',sprintf('Fitting (%i spectra) is in progress', other_n));
      k = 0;
      for ii=1:sz(2)
        fpars.yy = real(double(yy(:,ii)));
        fmin = @(x) sqrt(sum((fpars.yy - simple_fit(x,fpars)).^2));
        out_pars(1:4,ii) = fminsearch(fmin, [max(abs(fpars.yy)), B0, 20e-3, 0]);
        out_fit = simple_fit(out_pars(1:4,ii),fpars);
        %       out_pars(1,ii) = (fpars.yy'*out_fit)/(out_fit'*out_fit);
        %       out_errs(2:4,ii)=xx_err;
        %       out_errs(5,ii)=sqrt(sum((fpars.yy/out_pars(1,ii)-out_fit).^2))/sum(abs(out_fit));
        waitbar(k/other_n, h); k = k+1;
      end
    else
      for ii=1:sz(2)
        fpars.yy = yy(:,ii);
        fpars.PG(1:3,1) = out_pars(2:4, ii);
        [xx, xx_err]=cw_grad_min(@simple_fit, fpars);
        out_pars(2:4,ii) = xx;
        out_fit = simple_fit(xx,fpars);
        out_pars(1, ii) = (fpars.yy'*out_fit)/(out_fit'*out_fit);
        out_errs(2:4,ii)=xx_err;
        out_errs(5,ii)=sqrt(sum((fpars.yy/out_pars(1,ii)-out_fit).^2))/sum(abs(out_fit));
      end
    end
  catch err
    disp(['Fitting error: ', err.message]);
  end
end
warning on;
if ishandle(h), delete(h); end

% --------------------------------------------------------------------
function y = simple_fit(x, pars)

lw = epr_Lorentzian(pars.Bshf, 0, max(x(3), 0.0001));
y = real(conv2(lw(:,1)*exp(-1i*x(4)),pars.pattern,'same'));
y = abs(x(1))*interp1(pars.Bshf+x(2), y, pars.B);

% --------------------------------------------------------------------

function y1 = fix_baseline(y)
% option 1: do nothing
% y1 = y;
%option 2: simmetrize uzing derivative
deriv = diff(y);
OFF = mean(deriv); deriv = deriv - OFF;
intg = cumsum(deriv); y1 = [intg; intg(end)];