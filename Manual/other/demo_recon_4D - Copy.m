%% Generate equal angular linear angle layout of projections

fbp_struct.nAz = 10;
fbp_struct.nPolar = 10;
fbp_struct.nSpec = 14;
fbp_struct.imtype = iradon_GetFBPImageType('XYZB');
fbp_struct.MaxGradient = 3.024;
fbp_struct.deltaL = 5;
% fbp_struct.angle_sampling = 'UNIFORM_ANGULAR_FLIP';
fbp_struct.angle_sampling = 'UNIFORM_SPATIAL_FLIP';

[pars, pars_ext] = iradon_FBPGradTable(fbp_struct);

% assign parameters for projection generation
radon_pars.x = pars_ext.kx;
radon_pars.y = pars_ext.ky;
radon_pars.z = pars_ext.kz;
radon_pars.w = pars_ext.w;

%% Projection generation for the 8-spheres phantom

R = 2.2; % Radius of the big sphere
Rs = 0.5; % Radius of the small spheres
a = 1.3;  % offset value compared to the center
nBins = 64;
phan1 = struct('nBins', nBins, 'r', R, 'offset', [0,0,0]); % sphere data
phan2 = struct('nBins', nBins, 'r', Rs, 'offset', [0,0,0]); % sphere data
phan3 = struct('nBins', nBins, 'r', Rs, 'offset', [a,0,0]); % sphere data
phan4 = struct('nBins', nBins, 'r', Rs, 'offset', [-a,0,0]); % sphere data
phan5 = struct('nBins', nBins, 'r', Rs, 'offset', [0,a,0]); % sphere data
phan6 = struct('nBins', nBins, 'r', Rs, 'offset', [0,-a,0]); % sphere data
phan7 = struct('nBins', nBins, 'r', Rs, 'offset', [0,0,a]); % sphere data
phan8 = struct('nBins', nBins, 'r', Rs, 'offset', [0,0,-a]); % sphere data

radon_pars.size = 5;
P1 = radon_c2d_sphere(phan1, radon_pars);
P2 = radon_c2d_sphere(phan2, radon_pars);
P3 = radon_c2d_sphere(phan3, radon_pars);
P4 = radon_c2d_sphere(phan4, radon_pars);
P5 = radon_c2d_sphere(phan5, radon_pars);
P6 = radon_c2d_sphere(phan6, radon_pars);
P7 = radon_c2d_sphere(phan7, radon_pars);
P8 = radon_c2d_sphere(phan8, radon_pars);
P = 1.0*P1 + (2.0 - 1.0)*P2 + (1.8 - 1.0)*P3 ...
  + (1.6 - 1.0)*P4 + (1.4 - 1.0)*P5 + (1.2 - 1.0)*P6 ...
  + (0.5 - 1.0)*P7 + (0.0 - 1.0)*P8; %Note: it is suggested that the denstity range be [0 1]. zhiwei 11/8

clear P1 P2 P3 P4 P5 P6 P7 P8
%% generate 4D projections

R = 2.2; % Radius of the big sphere
Rs = 0.25; % Radius of the small spheres
a = 1.3;  % offset value compared to the center
nBins = 64;
% ksum field is used to determine the summing coefficient
phan{1}.par = struct('nBins', nBins, 'r', R, 'offset', [0,0,0], 'ksum', 0.0); % sphere data
phan{2}.par = struct('nBins', nBins, 'r', Rs, 'offset', [0,0,0], 'ksum', 1.0); % sphere data
phan{3}.par = struct('nBins', nBins, 'r', Rs, 'offset', [a,0,0], 'ksum', 1.0); % sphere data
phan{4}.par = struct('nBins', nBins, 'r', Rs, 'offset', [-a,0,0], 'ksum', 1.0); % sphere data
phan{5}.par = struct('nBins', nBins, 'r', Rs, 'offset', [0,a,0], 'ksum', 1.0); % sphere data
phan{6}.par = struct('nBins', nBins, 'r', Rs, 'offset', [0,-a,0], 'ksum', 1.0); % sphere data
phan{7}.par = struct('nBins', nBins, 'r', Rs, 'offset', [0,0,a], 'ksum', 1.0); % sphere data
phan{8}.par = struct('nBins', nBins, 'r', Rs, 'offset', [0,0,-a], 'ksum', 1.0); % sphere data

cos_alpha = cos(pars_ext.alpha);
tan_alpha = max(tan(pars_ext.alpha));
deltaB =  pars.data.FBP.MaxGradient * fbp_struct.deltaL / tan_alpha;
ReconSweep = pars.UnitSweep * pars.deltaH;

alpha_radon_pars.size = 5;

P = zeros(phan1.nBins, pars.nP);
for nphantom = 2:8
  for ii = 1:pars.nP
    alpha_radon_pars.x = radon_pars.x(ii);
    alpha_radon_pars.y = radon_pars.y(ii);
    alpha_radon_pars.z = radon_pars.z(ii);
    PP = radon_c2d_sphere(phan{nphantom}.par, alpha_radon_pars);
    x_scan = linspace(-ReconSweep(ii)/2, ReconSweep(ii)/2, nBins);
    lineshape = real(epri_lshape(x_scan, 0, [0.3 0.3], 1));
    P(:,ii) = P(:,ii) + conv(PP, lineshape', 'same');
  end
end


%% FBP multistage image reconstruction

PP = zeros(phan1.nBins, fbp_struct.nAz*fbp_struct.nPolar*fbp_struct.nSpec);
PP(:,pars.gidx) = P;

% convert serial projection layout into matrix form required
% for multistage reconstruction
Pela = reshape(PP, [size(P,1), fbp_struct.nAz, fbp_struct.nPolar, fbp_struct.nSpec]);

% Interpolate to uniform angular
switch fbp_struct.angle_sampling
  case {'UNIFORM_SPATIAL','UNIFORM_SPATIAL_FLIP'}
    Pela=iradon_InterpToUniformAngle(Pela, 'imgData');
end

% projection visualization
% MatrixGUI(phantom)

radon_pars.ELA =  fbp_struct;
recon_pars.size = 5;     % this par is ignored, radon_pars are used 
recon_pars.nBins = 64;  % this par is ignored, radon_pars are used
recon_pars.Filter = 'ram-lak';
recon_pars.FilterCutOff = 1.0;
recon_pars.Interpolation = 'spline';
recon_pars.InterpFactor = 2; % better 4
recon_pars.CodeFlag = 'MATLAB';
recon_pars.zeropadding = 1; % any number >= 1

% call the reconstruction program and display result
mat_recFXD = iradon_d2d_mstage(Pela, radon_pars, recon_pars);
ibGUI(mat_recFXD);

