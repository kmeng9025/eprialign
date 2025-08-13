%%
cd Z:\CenterMATLAB\3dparty\FPGApulse

%%
t2ns = 1;
t1us = 500*t2ns;

n = 500;
trep = t1us/10;

ff = 10;

pulse_timing = zeros(5*n, 3);

for ii=1:n
  pulse_timing(1+5*(ii-1),:) = [1, t2ns * 0 + (ii-1)*trep, t2ns*5 + (ii-1)*trep];
  pulse_timing(2+5*(ii-1),:) = [3, t2ns * 10 + (ii-1)*trep, t2ns*15 + (ii-1)*trep];
  pulse_timing(3+5*(ii-1),:) = [1, t2ns * 6 + (ii-1)*trep, t2ns*12 + (ii-1)*trep];
  pulse_timing(4+5*(ii-1),:) = [2, t2ns * 8 + (ii-1)*trep, t2ns*13 + (ii-1)*trep];
  pulse_timing(5+5*(ii-1),:) = [7, t2ns * 20 + (ii-1)*trep, t2ns*26 + (ii-1)*trep];
end

pulse_timing(:,2) = pulse_timing(:,2) * ff;
pulse_timing(:,3) = pulse_timing(:,3) * ff;

p = pgenpulseprog(pulse_timing, bin2dec('0000000000000000'));

% pgenPlotPulses(p, 1)

%% Send to device

tic
obj = serial('COM3', 'BaudRate', 128000);
fopen(obj);
% pgensendprog(obj, p, 0)
pgensendprog_mod(obj, p, 0)
fclose(obj);
toc
