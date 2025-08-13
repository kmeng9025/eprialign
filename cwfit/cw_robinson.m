function [y_in, y_out] = cw_robinson(x, x_not, R2, mod_omega, mod_amp, max_harmonic)
% [y_in y_out] = cw_robinson(x, x_not, R2, mod_omega, mod_amp, max_harmonic)
% x_not, the center of the line in gauss
% R2 = 1/T2, the spin-spin relaxation rate in gauss.
% mod_omega, the modulation frequency in gauss
% mod_amp, the peak to peak modulation amplitude in gauss. 

% the amplitude of this function is identical to the lor_fm up to the
% coefficient 4/mod_amp per harmonic

% Shift to center, make a column
x = x(:) - x_not;

n=max_harmonic; % cutoff harmonic

a0 = x+1i*R2;
hm2 = mod_amp*mod_amp/16;
M = length(x);
GN=zeros(M*2, n); % allocate n harmonics and plus/minus
a_pm = [a0+n*mod_omega; a0-n*mod_omega]; % [ap; am]
GN(:, n) = a_pm.*(1 + sqrt(1-4*hm2./(a_pm.^2)))/2;

% create g+/g-
for r=n-1:-1:1
    ar = [ a0+r*mod_omega; a0-r*mod_omega];
    GN(:,r) = ar-hm2./GN(:, r+1);
end

Y0=1./(a0-hm2*(1./GN(1:M,1)+1./GN(M+1:end,1)));
Y=0*GN;
Y(:,1)=-mod_amp./GN(:,1).*[Y0;Y0]/4;

for r=2:n
    Y(:,r)=-mod_amp./GN(:,r-1).*Y(:, r-1)/4;
end

Y = Y(:, 1:max_harmonic);

Yminus=Y(1:M, :);
Yplus =Y(M+1:end, :);

y_in=-(Yplus+Yminus)/pi/2;
y_out=-1i*(Yplus-Yminus)/pi/2;

