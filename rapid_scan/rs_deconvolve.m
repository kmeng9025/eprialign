function [x_ss,ss,rsi]=rs_deconvolve(rs, sweep, freq, dt, M, MaxCycles)

% Average projections
Npt  = size(rs, 1);           % trace length
Ntr  = size(rs, 2);           % number of traces
P  =  1/freq;                 % Scan period
N  = fix(Npt*dt/P);           % number of cycles in the trace

if ~exist('M', 'var'), M=2^14; end
if exist('MaxCycles','var'), N = min(N,MaxCycles); end

x = dt * (0:Npt-1);
x_ss = (0:(M-1))'/M * sweep*2;

rsi = zeros(M, Ntr);

for ii=1:Ntr
  for jj=1:N
    pIdx = x >= ((jj-1) * P - 2*dt) & x < (jj * P + 2*dt);
    
    x1 = x(pIdx)';
    x2 = linspace((jj-1) * P, jj * P - dt, M)';
    rsi_trace = interp1(x1,rs(pIdx,ii),x2);
    rsi_trace(isnan(rsi_trace)) = 0;
    rsi(:,ii)=rsi(:,ii)+rsi_trace;
  end
end
rsi=rsi / N;

use_baseline = min(100, M/20);
bl_idx = [1:use_baseline,M-use_baseline:M];
rsi = rsi -  repmat(mean(rsi(bl_idx, :), 1), [M, 1]);

% Deconvolution
h=fft(rsi, [], 1);

% gamma_one = bmagn/planck*2.0023 * 1e-4; % Hz/G
% b = gamma_one * 2*sweep * 2 * pi * freq; 
% gamma = gamma_one * 4 *pi;
gamma = 3.5217e+007;

b2 = 2 * gamma * sweep * freq; 

w=(-M/2+1:M/2)'*2*pi*freq;
B = exp(-1i*w.^2/b2); % analytical function
B(1:fix(M/2),:) = conj(B(1:fix(M/2),:));
B = fftshift(B);

% figure(3); clf; hold on
% plot(w, real(B), w, imag(B))
% plot(w, real(h(:,1)), w, imag(h(:,1)))

hh = h./B(:, ones(Ntr,1));
ss=ifft(hh, [], 1); % slow scan spectrum

% figure(3); clf; 
% subplot(2,1,1); hold on
% plot(x, real(rs), x, imag(rs))
% ax = axis; for ii=1:N, plot(P*ii*[1,1], ax([3,4]), 'r-'); end
% ylabel 'Intensity, a.u.'
% axis tight
% subplot(2,1,2); hold on
% plot(x_ss,real(ss(:,1)));
% plot(x_ss,imag(ss(:,1)), 'g'); 
% plot(x_ss,real(rsi(:,1)), ':b'); 
% plot(x_ss,imag(rsi(:,1)), ':g'); 
% axis tight; 
% text(0.08, 0.9, sprintf('Sweep: %4.2f G',sweep),'units', 'normalized')
% text(0.08, 0.8, sprintf('Freq:  %4.2f Hz',freq),'units', 'normalized')
% xlabel 'Field offset [G]'
% ylabel 'Intensity, a.u.'
