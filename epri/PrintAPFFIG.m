% function PrintAPFFIG(fig, APF, opt)
%  fig - figure number
%  APF - [amplitude; phase; frequency_offset] nx3 matrix
%  opt - struct
%     center_frequency   center frequency
%     lorentz_frequency  center frequency of Lorentzian
%     lorentz_Q          quality factor of Lorentzian

function PrintAPFFIG(fig, APF, opt)

idx =  APF(:, 1) >= 0.01*max(APF(:, 1));
APF = APF(idx, :);

[aaa, idx] = sort(APF(:,3));
APF = APF(idx,:);

center_frequency = safeget(opt, 'center_frequency', 250.0);
APF(:,3) = APF(:,3) + center_frequency;

off3dB = find(APF(:,1) > max(APF(:,1))*0.5);
off3dBmin = APF(min(off3dB),3);
off3dBmax = APF(max(off3dB),3);

if isfield(opt, 'lorentz_Q')
  lorentz_frequency = safeget(opt, 'lorentz_frequency', 250.0);
  lorentz_Q = safeget(opt, 'lorentz_Q', 6.6);
  c_profile = lshape(APF(:,3),lorentz_frequency,lorentz_frequency/lorentz_Q);
  draw_lorentz = 1;
else
  draw_lorentz = 0;
end

normalize_amplitude = safeget(opt, 'normalize_amplitude', 0);
if normalize_amplitude
  APF(:,1) = APF(:,1)./max(APF(:,1));
end

yy_func = @(xx, x)xx(1)*lshape(x,xx(2),xx(3),0, .4);
yy_diff = @(xx, x, data)sqrt(sum((data - yy_func(xx, x)).^2));

xx = [max(APF(:,1)) * 6, 250, 6];
XX = fminsearch(yy_diff, xx, [], APF(:,3), APF(:,1));

figure(1)
clf
subplot(2,1,1)
plot(APF(:,3), APF(:,1), '.-'); hold on
xlabel('Frequency'); ylabel('Amplitude');
if draw_lorentz
  plot(APF(:,3), yy_func(XX, APF(:,3)), 'r-'); hold on
  title(sprintf('BW(3dB) = %5.3g MHz, BW(nu/Q) = %5.3g MHz', off3dBmax-off3dBmin, lorentz_frequency/lorentz_Q));
  plot(APF(:,3), c_profile*max(APF(:,1))/max(c_profile), 'm')
  legend({'';sprintf('Lorentz Q=%4.1f', lorentz_Q)})
else
  title(sprintf('BW(3dB) = %5.3g MHz', off3dBmax-off3dBmin));
end
axis tight
subplot(2,1,2)
plot(APF(:,3), APF(:,2), '.-');
xlabel('Frequency'); ylabel('Phase')
axis tight
