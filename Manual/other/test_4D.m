fbp_struct.nAz = 10;
fbp_struct.nPolar = 10;
fbp_struct.nSpec = 14;
fbp_struct.imtype = 1;


% [pars, pars_ext] = td_GetFBPGradientTable(fbp_struct);


aalpha = pi/2+pi/fbp_struct.nSpec/2-pi/fbp_struct.nSpec*(1:fbp_struct.nSpec);

phantom = zeros(64,14,10,10);

dB = 1; % G
lw = 0.15;

% lw shoud be smaller than 25% of dB and larger than 5%

for ii=1:14
  cos_alpha = cos(aalpha(ii));
  specx = linspace(-dB/2/cos_alpha, dB/2/cos_alpha, 64);
  specy = epri_lshape(specx, 0, [0, lw]);
  phantom(:, ii, :, :) = repmat(specy', [1,10,10])/cos_alpha;
end

% MatrixGUI(phantom)

radon_pars.ELA =  fbp_struct;
radon_pars.size = dB;
recon_pars.nBins = 64;
recon_pars.Filter = 'ram-lak';
recon_pars.FilterCutOff = 0.5;
recon_pars.Interpolation = 'spline';
recon_pars.InterpFactor = 4;
recon_pars.CodeFlag = 'MATLAB';
mat_recFXD = iradon_d2d_mstage(phantom, radon_pars, recon_pars);

ibGUI(mat_recFXD)

%%

raw_info.deltaH = dB;
fit = processing_struct.fit;
[fit_data] = rs_spectral_fit(mat_recFXD, raw_info, fit);



