function SNR = epri_SNR(data, noise_idx, signal_idx)

Baseline = data(noise_idx);
data = data - mean(Baseline);

if ~exist('signal_idx', 'var'), signal_idx = 1:length(data); end

Signal = max(abs(data(signal_idx)));
Noise = std(data(noise_idx));

SNR = Signal/Noise;