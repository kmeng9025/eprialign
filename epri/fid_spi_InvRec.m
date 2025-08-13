% FinalImage = fid_spi_InvRec(file_name, file_suffix, output_path, fields)
%   file_name:     full name of the file
%   file_suffix:   string to be added before extension
%   output_path:   the path where to store file, name will be preserved,
%                  suffix will be added
%   fields:        structure with fields td, fft, prc, clb etc
% FinalImage = ese_fbp({file_names}, file_suffix, output_path, fields)
%   {file_names}"  a cell array of filenames to be summed before
%                  processing
% FinalImage = ese_fbp(Loaded_Image, file_suffix, output_path, fields)
%   Loaded_Image: a struct with raw_info, mat and mat_bl fields

function FinalImage = fid_spi_InvRec(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');
is_export_proj = strcmp(safeget(fields.prc, 'export_prj','yes'),'yes');
is_save_data = strcmp(safeget(fields.prc, 'save_data','yes'),'yes');

fnames = epri_filename(file_name, file_suffix, output_path);

load_opt.Modality = 'PULSESPI';
load_opt.Sequence  = 'FIDInvRec';
load_opt.SPI = fields.spi;

% Load data
[out,FinalImage] = epri_load_for_processing(file_name, load_opt);

rec_info.td = fields.td;
rec_info.fft = fields.fft;
% rec_info.fft.FOV = fields.rec.Size*fields.fbp.MaxGradient*2.802;
rec_info.rec = fields.spi;
rec_info.rec = copy_fields(rec_info.rec, fields.rec);
rec_info.rec.Enabled = iff(strcmp(safeget(fields.prc, 'recon_data','yes'),'yes'), 'on', 'off');
rec_info.prc = fields.prc;

% Pulse data do not have Hardware calibration
fields.clb.ampHH = 1;

if is_recon_data
  if isfield(rec_info.rec, 'recon_echos') && ~isempty(rec_info.rec.recon_echos)
    list_of_echoes = rec_info.rec.recon_echos;
    if isfield(rec_info.td, 'use_echos') && ~isempty(rec_info.td.use_echos)
      idx = zeros(length(FinalImage.raw_info.tau2), 1);
      idx(rec_info.td.use_echos) = 1;
      rec_info.td.use_echos = find(idx(list_of_echoes));
    end
    FinalImage.raw_info.T = FinalImage.raw_info.T(list_of_echoes);
  else
    list_of_echoes = 1:size(out.mat, 2);
  end
  yyy = out.mat - out.mat_bl;
   
  FinalImage.Raw = [];
  % run reconstruction for every echo --------------------------------------
  image_str = cell(numel(list_of_echoes), 1);
  for echo_n = list_of_echoes
    image_str{echo_n} = ['Image ', num2str(echo_n)];
  end
  
  FinalImage.rec_info = rec_info;
  
  tm = FinalImage.raw_info.t_ax*1E6 + rec_info.td.acq_window_start*1E-3; % us
  SPIdelay = rec_info.rec.SPIdelay * 1E-3*ones(length(list_of_echoes), 1);
  
  gidx = FinalImage.raw_info.gidx;
  Gmax = FinalImage.raw_info.MaxGradient;
  
%   [yyy,st_td] = kv_baseline_correction(yyy, rec_info.td);

  
  for pnt_n=1:length(list_of_echoes)
    [~, idx] = min((tm-SPIdelay(pnt_n)).^2);
    tp = tm(idx);
    FinalImage.rec_info.Size = FinalImage.raw_info.Dim/2/tp/Gmax/2.8.*(1+2./FinalImage.raw_info.Dim);
    fprintf('Selected point: %i (%f us)\n', idx, SPIdelay(pnt_n));
    yTP = zeros(FinalImage.raw_info.Dim);
    
    for ii=1:size(yyy, 3)
      yyy(:, pnt_n, ii) = smooth(yyy(:, pnt_n, ii), 7);
    end
    yTP(gidx) = squeeze(yyy(idx, pnt_n ,:));

%     figure(100); clf;
%     for ii=2913-5:2913+5 % 1:size(mat, 3)
%       trace = squeeze(yyy(idx-50:end, pnt_n, ii));
%       plot(tm(idx-50:end), real(trace), tm(idx-50:end), imag(trace));
%       text(0.1, 0.9, sprintf('%d : %d', pnt_n, ii), 'Units', 'normalized');
%       axis tight
%       pause
%     end

%     d_idx = yTP ~= 0;
%     bl = d_idx & imerode(d_idx, epr_strel('sphere', 2));
%     im_bl = mean(yTP(bl(:)));
%     yTP(d_idx) = yTP(d_idx) - im_bl;
    
    yTP = yTP * exp(1i*rec_info.td.phase_zero_order*pi/180);
    %    right_phase = rec_info.td.phase_zero_order*pi/180*ones(length(use_echos),sz(3));
    
    %   yTP = squeeze(sum(yy(idx-1:idx+1,:,:,:),1));
    %
    %   disp(handles.mat_info.Dim/2/tp/Gmax/2.8)
    
    %filtering
    data_filter = safeget(fields.td, 'filt_before', 'none');
%     data_filter = 'han';
    switch data_filter
      case {'ham', 'han', 'cos'}
        sz = size(yTP);
        f1 = apowin(data_filter, sz(1));
        f2 = apowin(data_filter, sz(2));
        f3 = apowin(data_filter, sz(3));
        yF = repmat(f1, [1, sz(2), sz(3)]).*...
          permute(repmat(f2, [1, sz(1), sz(3)]), [2,1,3]).*...
          permute(repmat(f3, [1, sz(1), sz(2)]), [3,2,1]);
        yTP = yTP.*yF;
      otherwise
        yF = ones(size(yTP));
    end
    
%     ImageBrowserGUI(real(yTP))
   
    % reconstruction
    switch 'as_is'
      case 'as_is'
        yTPzero = zeros(FinalImage.raw_info.Dim*3);
        yTPzero(FinalImage.raw_info.Dim(1)+1:2*FinalImage.raw_info.Dim(1),...
          FinalImage.raw_info.Dim(2)+1:2*FinalImage.raw_info.Dim(2),...
          FinalImage.raw_info.Dim(3)+1:2*FinalImage.raw_info.Dim(3)) = yTP;

        FinalImage.Raw(:,:,:,pnt_n) = (fftshift(ifftn(ifftshift(yTPzero))));
        
        FinalImage.Raw = abs(FinalImage.Raw) .* sign(real(FinalImage.Raw));
        %      handles.mat_rec_info
        %     case 'interpolation'
        %       FXD = real(fftshift(ifftn(ifftshift(yTP))));
        %       zf_size = handles.mat_info.Dim/2/tp/Gmax/2.8;
        %
        %       scale = mean(handles.mat_rec_info.SizeObj ./ zf_size);
        %
        %       x = handles.mat_info.Dim(1)/2 + 0.5 + ...
        %         ((1 : handles.mat_info.Dim(1)) - handles.mat_info.Dim(1)/2 - 0.5) * scale;
        %       y = handles.mat_info.Dim(2)/2 + 0.5 + ...
        %         ((1 : handles.mat_info.Dim(2)) - handles.mat_info.Dim(2)/2 - 0.5) * scale;
        %       z = handles.mat_info.Dim(3)/2 + 0.5 + ...
        %         ((1 : handles.mat_info.Dim(3)) - handles.mat_info.Dim(3)/2 - 0.5) * scale;
        %       [xi,yi,zi] = meshgrid(y,x,z);
        %       handles.mat_recFXD(:,:,:,pnt_n)  = interp3(FXD, xi,yi,zi)*scale^3;
        %       handles.mat_rec_info.Dim = handles.mat_info.Dim;
        %     case 'zero_filling'
        %       zf_size = round(smooth_factor * 2*tp*Gmax*handles.mat_rec_info.SizeObj);
        %       yTPzero = zeros(handles.mat_info.Dim*3);
        %       yTPzero(handles.mat_info.Dim(1)+1:2*handles.mat_info.Dim(1),...
        %         handles.mat_info.Dim(2)+1:2*handles.mat_info.Dim(2),...
        %         handles.mat_info.Dim(3)+1:2*handles.mat_info.Dim(3)) = yTP;
        %       % ShowSlice3DFIG(100, real(yTP), [])
        %
        %       %       x = ((1:handles.mat_info.Dim(1)*3)-handles.mat_info.Dim(1)*3/2 - 0.5)/2/tp/Gmax/3;
        %       %       y = ((1:handles.mat_info.Dim(2)*3)-handles.mat_info.Dim(2)*3/2 - 0.5)/2/tp/Gmax/3;
        %       %       z = ((1:handles.mat_info.Dim(3)*3)-handles.mat_info.Dim(3)*3/2 - 0.5)/2/tp/Gmax/3;
        %       handles.mat_recFXD(:,:,:,pnt_n) = real(fftshift(ifftn(ifftshift(yTPzero))));
        %       handles.mat_rec_info.SizeObj = handles.mat_info.Dim/2/tp/Gmax/3;
    end
    %   if usepause, pause; else drawnow; end
  end
  
  baseline_stat(yyy, fields.td);
  FinalImage.Size = FinalImage.rec_info.rec.Size;
  
  if strcmpi(safeget(fields.fft, 'fft_export_clearence', 'no'), 'yes')
    FinalImage.Clearence = safeget(dsc, 'export_clearence', []);
  end
  
  % Save raw image
  if is_save_data
    epri_create_directory(fnames.path);
    s.file_type    = 'Image_v1.1';
    s.raw_info     = FinalImage.raw_info;
    s.traces       = single(yyy);
%     s.projections  = single(y);
    s.mat_recFXD   = single(FinalImage.Raw);
    s.rec_info = FinalImage.rec_info;
    s.pO2_info = fields.clb;
    save(fnames.raw_file,'-struct','s');
    fprintf('File %s is saved.\n', fnames.raw_file);
  end
elseif exist('s', 'var')
  FinalImage.Raw = s.mat_recFXD;
  FinalImage.rec_info = s.rec_info;
end

if ndims(FinalImage.Raw) ~=4, is_fit_data = false; end
if is_fit_data
  [FinalImage.fit_data] = epri_recovery_fit(FinalImage.Raw, FinalImage.raw_info.T(:)'*1E6, fields.fit);
  
  if is_save_data
    epri_create_directory(fnames.path);
    s1.file_type    = 'FitImage_v1.1';
    s1.source_image = fnames.raw_file;
    s1.p_image = fnames.p_file;
    s1.raw_info     = FinalImage.raw_info;
    s1.fit_data     = FinalImage.fit_data;
    s1.rec_info     = FinalImage.rec_info;
    s1.pO2_info     = fields.clb; %#ok<STRNU>
    save(fnames.p_file,'-struct','s1');
    fprintf('File %s is saved.\n', fnames.p_file);
  end
  
  % Load fit results for display
  [FinalImage.Amp, FinalImage.T1, FinalImage.R1, FinalImage.Mask, FinalImage.Error, eR1] = LoadFitPars(FinalImage.fit_data, {'Amp','T1','R1','Mask','Error', 'Error_R1'});
  FinalImage.pO2 = epr_T2_PO2(FinalImage.T1, FinalImage.Amp, FinalImage.Mask, fields.clb);
  Torr_per_mGauss = safeget(fields.clb, 'Torr_per_mGauss', 1.84);
  FinalImage.Error_O2 = eR1/pi/2/2.802*1000*Torr_per_mGauss;
  %       Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
  Q_correction = 1;
  FinalImage.Amp = FinalImage.Amp * Q_correction/ fields.clb.amp1mM;
else
  FinalImage.fit_data = [];
end

FinalImage.ImageName = fnames.raw_file;


% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end



%---------------------------------------------------------------
function baseline_stat(y, st_td)
figure(4); clf;
sz = size(y);
pst = epr_CalcAxesPos(sz(2), 1, [0.06 0.0005], [0.04 0.05]);

for tau = 1:sz(2)
  h(tau) = axes('Position', pst(tau,:)); hold on
  for ii=1:50:size(y,3), plot(1:size(y,1),real(y(:,tau,ii)),1:size(y,1),imag(y(:,tau,ii))); end
  if tau == 1, title('After base line correction'); end
  axis('tight'); xlabel('Points');
end
