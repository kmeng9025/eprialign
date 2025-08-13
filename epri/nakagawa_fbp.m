function FinalImage = nakagawa_fbp(file_name, file_suffix, output_path, fields)

% Load data

[ax,y,dsc]=kv_d01read(file_name, 'Return', {{'params_Out1', 'G/cm'}, {'params_Out2', 'G/cm'}, {'params_SweepWidth',''}});
recon_data = real(y(:,2:end));
reference_data = real(y(:,1));
dsc.general_name
image_type = safeget(dsc, 'exp_info_template', fields.fbp.image);
disp(size(y))
% code = 'SB14T13G14S1P3Z-1O0N0';

FBP = epri_DecodeImageType(image_type);
[out, suplementary_out] = iradon_FBPGradTable(FBP);

ref_data = real(y(:,1));
recon_data = real(y(:,2:end));

nprj = size(recon_data, 2);

if strcmpi(safeget(fields.ppr, 'launch_GUI', 'yes'), 'YES')
  DeconvolveGUI(y);
end

% process data (deconvolve)
if strcmpi(safeget(fields.ppr, 'deconvolve', 'yes'), 'YES')
  pars.function = safeget(fields.ppr, 'function', 'Gaussian');  % Gaussian or Fermi
  pars.fwhm = safeget(fields.ppr, 'fwhh', 15); % for Gaussian more is more noisy but higher resolution
  pars.radius = safeget(fields.ppr, 'radius', 15); % for fermi  more is more noisy but higher resolution
  pars.roll   = safeget(fields.ppr, 'roll', 6); % for Fermi make cutoff sharper
  offset = safeget(fields.ppr, 'offset', 0);
  
  recon_data1 = epr_deconvolve([],recon_data,[],circshift(reference_data,offset),pars);
  
  ref_data1 = epr_deconvolve([],real(y(:,1)),[],circshift(reference_data,offset),pars);
else
  bl = mean(mean(ref_data)); rd = ref_data - bl;
  ref_data1 = cumsum(rd);
  
  bl = mean(recon_data);
  rd = recon_data - repmat(bl, size(recon_data, 1), 1);
  recon_data1 = cumsum(rd);
end

opt.FigFFT = 4;
xx  = repmat(1:size(recon_data1), nprj,1)';
idx{1} = find(1:nprj);
prj_stat(xx, recon_data1, idx, opt);

% prepare data for reconstruction

[~, B0idx] = max(ref_data1);
B0 = ax.x(B0idx);

sweep = 2*max(abs(ax.x - B0))*1.0;
np   = safeget(fields.rec, 'nBins', 250);       % number of points
nprj = size(recon_data1,2); % number of projections

% shift center to B0
FLD = linspace(B0-sweep/2, B0+sweep/2, np);
projection_data = zeros(np, nprj);

for ii=1:nprj
  projection_data(:, ii) = interp1(ax.x, recon_data1(:, ii), FLD, 'spline', 0);
end


% reconstruct data

angles = suplementary_out.phi*180/pi;
data_for_recon = projection_data;
% data_for_recon = dec_data;
FinalImage.Raw = iradon(data_for_recon, angles,'spline', 'Ram-Lak', 1);

if ndims(FinalImage.Raw) == 2
  FinalImage.Raw = repmat(FinalImage.Raw,[1,1,2]);
end


function prj_stat(x,y,idx,opt)
figure(safeget(opt, 'FigFFT', 4)); clf; hold on

nSet = min(length(idx), 7);
pst=epr_CalcAxesPos(nSet, 1, [0.06 0.0005], [0.04 0.05]);

h = zeros(nSet, 1);
for ii = 1:nSet
  h(ii) = axes('Position', pst(ii,:)); hold on
  xset = x(:, idx{ii});
  yset = y(:, idx{ii});
  for jj=1:size(yset, 2)
    plot(xset(:,jj), real(yset(:,jj)), 'b');
    %     plot(rx,imag(ry(:,ii,jj)), 'g');
  end
  ystat = mean(sum(yset));
  sstat_dev = std(sum(yset));
  text(0.05, 0.85, sprintf('I=%5.3f(%4.3f)', ystat, sstat_dev), 'units', 'normalized')
  axis tight
end
set(h(1:end-1), 'XTickLabel', '');
set(h, 'Box', 'on');
xlabel(h(end), '[G]')
