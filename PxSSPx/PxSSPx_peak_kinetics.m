function res = PxSSPx_peak_kinetics(data, peak, opts)

[~,idx1] = min(abs(peak.x - (peak.OFF - peak.A/2)));
[~,idx3] = min(abs(peak.x - (peak.OFF + peak.A/2)));
[~,idx2] = min(abs(peak.x - (peak.OFF)));

range_1_3 = -100:100;
range_2 = -100:100;

res.L1 = sum(data.y(idx1+range_1_3,:), 1);
res.L2 = sum(data.y(idx2+range_2,:), 1);
res.L3 = sum(data.y(idx3+range_1_3,:), 1);
res.time = data.time;



