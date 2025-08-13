function sampling = epri_tracelength2sampling(tracelength, nPts)
samplings = [2e-9, 4e-9, 5e-9, 10e-9, 20e-9, 40e-9, 50e-9, 100e-9, 200e-9, 400e-9, 500e-9,...
  1e-6, 2e-6, 4e-6, 5e-6, 10e-6, 20e-6, 40e-6, 50e-6, 100e-6];
sampling = zeros(length(tracelength), 1);
for ii=1:length(tracelength)
  idx = find(tracelength(ii)/nPts <= samplings, 1, 'first');
  sampling(ii) = samplings(idx);
end