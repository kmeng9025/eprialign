% FinalImage = fid_spi(file_name, file_suffix, output_path, fields)
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

function FinalImage = fid_spi2(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');
is_export_proj = strcmp(safeget(fields.prc, 'export_prj','yes'),'yes');
is_save_data = strcmp(safeget(fields.prc, 'save_data','yes'),'yes');

fnames = epri_filename(file_name, file_suffix, output_path);

load_opt.Modality = 'PULSESPI';
load_opt.SPI = fields.spi;
load_opt.Sequence = 'FID';
[out,FinalImage] = epri_load_for_processing(file_name, load_opt);
SPI = FinalImage.raw_info.data.SPI; % SPI structure as loaded from file
FinalImage.raw_info.data.Sequence = 'FID';

rec_info.td = fields.td;
rec_info.fft = fields.fft;
% rec_info.fft.FOV = fields.rec.Size*fields.fbp.MaxGradient*2.802;
rec_info.rec = fields.spi;
rec_info.rec = copy_fields(rec_info.rec, fields.rec);
rec_info.rec.Enabled = iff(strcmp(safeget(fields.prc, 'recon_data','yes'),'yes'), 'on', 'off');
rec_info.prc = fields.prc;

% Pulse data do not have Hardware calibration
fields.clb.ampHH = 1;

if strcmp(safeget(rec_info.td, 'off_res_baseline', 'yes'), 'yes') && ~isempty(out.mat_bl)
  yyy = out.mat - out.mat_bl;
else
  yyy = out.mat;
end

if is_recon_data 
  FinalImage.Raw = [];
  FinalImage.rec_info = rec_info;
 
  gidx = FinalImage.raw_info.gidx(FinalImage.raw_info.gidx ~= 0);
  Gmax = FinalImage.raw_info.MaxGradient;
  
  tm = FinalImage.raw_info.t_ax*1E6 + rec_info.td.acq_window_start*1E-9*1E6; % us
  Size = FinalImage.rec_info.rec.Size; %cm
  Dim  = FinalImage.raw_info.Dim; Dim = Dim(1);
  kcomp = safeget(FinalImage.rec_info.rec, 'kcomp', 1.0);
  dG = 2 * Gmax / Dim;
  SPIdelay = kcomp / (Size * dG *2.802e6) * 1e6; %us
  
  pnt_n = 1;
  [~, idx] = min((tm-SPIdelay).^2);
  FinalImage.rec_info.Size = Size;
  fprintf('Selected point: %i(%i) (%f us)\n', idx, length(tm), SPIdelay)
  yTP = zeros(FinalImage.raw_info.Dim);
  
  for ii=1:size(yyy, 3)
    yyy(:, pnt_n, ii) = datasmooth(yyy(:, pnt_n, ii), 5);
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
  % reconstruction
  switch 'czt'
    case 'czt'
      MM = safeget(rec_info.rec, 'Sub_points', 64);
      Size = safeget(rec_info.rec, 'Size', 3.0);
      FinalImage.rec_info.rec.Size = Size;

      SPI_range = rec_info.rec.SPIdelay*1e-3;
      SPIdelay = linspace(min(SPI_range), max(SPI_range), 8);
      fprintf('Delays %f\n',SPIdelay);
      FinalImage.raw_info.T = SPIdelay * 1e-6;
 
      FinalImage.Raw = zeros(MM,MM,MM,length(SPIdelay));
      for ii=1:length(SPIdelay)
        [tp, factor(ii)] = find_tp(SPIdelay(ii), MM, dG, Size);
        [~, idx] = min((tm-tp).^2);
        yTP = zeros(FinalImage.raw_info.Dim);
        yTP(gidx) = squeeze(yyy(idx, pnt_n ,:));
        res = myczt(yTP, MM, factor(ii));
        FinalImage.Raw(:,:,:,ii) = res(:,:,:);
      end
      
    case 'as_is'
      yTPzero = zeros(FinalImage.raw_info.Dim*3);
      yTPzero(FinalImage.raw_info.Dim(1)+1:2*FinalImage.raw_info.Dim(1),...
        FinalImage.raw_info.Dim(2)+1:2*FinalImage.raw_info.Dim(2),...
        FinalImage.raw_info.Dim(3)+1:2*FinalImage.raw_info.Dim(3)) = yTP;
      FinalImage.Raw(:,:,:,pnt_n) = real(fftshift(ifftn(ifftshift(yTPzero))));
      FinalImage.Size = FinalImage.rec_info.rec.Size;
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
  
  
  A = abs(squeeze(max(yyy,[],3)));
  B = squeeze(sum(sum(sum(FinalImage.Raw, 1),2),3));
  figure; plot(tm,A/max(A),SPIdelay,B/max(B),SPIdelay, factor/2);grid on
  legend({'Data Integral', 'Image integral', 'factor/2'});
  
  if strcmpi(safeget(fields.fft, 'fft_export_clearence', 'no'), 'yes')
    FinalImage.Clearence = safeget(dsc, 'export_clearence', []);
  end
  
  % Image postprocessing
  fields_img = safeget(fields, 'img', []);
  mirror_image = safeget(fields_img, 'mirror_image', [0,0,0]);
  if mirror_image(1), FinalImage.Raw = flip(FinalImage.Raw, 1); end
  if mirror_image(2), FinalImage.Raw = flip(FinalImage.Raw, 2); end
  if mirror_image(3), FinalImage.Raw = flip(FinalImage.Raw, 3); end

  FinalImage.Dim4Unit = 'T2*';
  FinalImage.Size = FinalImage.rec_info.rec.Size;

  % Save raw image
  if is_save_data
    epri_create_directory(fnames.path);
    s.file_type    = 'Image_v1.1';
    s.raw_info     = FinalImage.raw_info;
%     s.traces       = single(yyy);
%     s.projections  = single(y);
    s.mat_recFXD   = single(FinalImage.Raw);
    s.rec_info = FinalImage.rec_info;
    s.pO2_info = fields.clb;
    save(fnames.raw_file,'-struct','s');
    fprintf('File %s is saved.\n', fnames.raw_file);
  end
  
  FinalImage.ImageName = fnames.raw_file;
elseif exist('s', 'var')
  FinalImage.Raw = s.mat_recFXD;
  FinalImage.rec_info = s.rec_info;
  is_recon_data = 1;
end

FinalImage.ImageName = fnames.raw_file;
if is_fit_data
  [FinalImage.fit_data] = epri_decay_fit(FinalImage.Raw, FinalImage.raw_info.T(:)'*1E6, fields.fit);
  FinalImage.fit_data.Parameters = {'Amplitude'; 'T2*'};
  
  [FinalImage.Amp, FinalImage.T2star, FinalImage.Mask, FinalImage.Error, FinalImage.Error_T2] = LoadFitPars(FinalImage.fit_data, {'Amp','T2','Mask','Error', 'Error_T2'});
  FinalImage.pO2 = epr_T2star_PO2(FinalImage.T2star, FinalImage.Amp, FinalImage.Mask, fields.clb);
  %       Q_correction = sqrt(s1.mat_pO2_info.Qcb/s1.mat_pO2_info.Q);
  Q_correction = 1;
  FinalImage.Amp = FinalImage.Amp * Q_correction/ fields.clb.amp1mM;
  
  if is_save_data
    epri_create_directory(fnames.path);
    s1.file_type    = 'FitImage_v1.1';
    s1.source_image = fnames.raw_file;
    s1.p_image      = fnames.p_file;
    s1.raw_info     = FinalImage.raw_info;
    s1.fit_data     = FinalImage.fit_data;
    s1.rec_info     = FinalImage.rec_info;
    s1.pO2_info     = fields.clb; 
    save(fnames.p_file,'-struct','s1');
    fprintf('File %s is saved.\n', fnames.p_file);
  end  
else
  FinalImage.fit_data = [];
end

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end

% --------------------------------------------------------------------
function  [z] = czt3(x,m,factor)

b = (1 - 1/factor)/2;
w=exp(-1j*2*pi*(1-2*b)/m);
a=exp(1j*(2*pi*b+pi));

% 3D chirp z-transform using matlab czt function.
% x = complex 3d k-space matrix
% m = number of output points - a 3D matrix
% w and a are the same as described for czt
% Ex: To zoom-in 125% symmetrically:
% b=(1-0.8)/2; m=19; w=exp(-j*2*pi*(1-2*b)/m); a=exp(j*(2*pi*b));
% For FFTSHIFT, use: a=exp(j*(2*pi*b+pi));
% [z] = czt3(x,m,w,a);
% Different magnifications can be specified as 3 element vectors of m, w and a.

% ensure that m, w and a are 3 element vectors.
if (max(size(m))==1), m=[m,m,m]; end
if (max(size(w))==1), w=[w,w,w]; end
if (max(size(a))==1), a=[a,a,a]; end
n=size(x);

% Procedure: reshape x to a 2D matrix, call czt to transform one dimension.
% Repeat the same procedure for other axes by permuting the axes.
% first dimension
z = reshape(x,n(1),[]);
z = czt(z,m(1),w(1),a(1));
z = reshape(z,m(1),n(2),n(3));

% second dimension
z = reshape(permute(z,[2 3 1]),n(2),[]);
z = czt(z, m(2),w(2),a(2));
z = reshape(z,m(2), n(3), m(1));
z = permute(z, [3 1 2]);

% third dimension
z = reshape(permute(z,[3 1 2]),n(3),[]);
z = czt(z, m(3), w(3), a(3));
z = reshape(z,m(3), m(1), m(2));
z = permute(z,[2 3 1]);

function final = myczt(y, MM, factor)

if factor >= 1.0
      final = abs(czt3(y, MM, factor)) / factor^3;
      fprintf('factor=%5.3f MM2 = %i\n', factor, MM);
else
      MM2 = floor(MM *factor + 0.5);
      fprintf('factor=%5.3f MM2 = %i\n', factor, MM2);
      new_factor = MM2/MM;
      res = czt3(y, MM2, 1);
      final = zeros(MM,MM,MM);
      idx1 = floor((MM - MM2) / 2)+1;
      idx2 = idx1 + MM2-1;
      final(idx1:idx2,idx1:idx2,idx1:idx2) = abs(res)/new_factor^3;
end

function [new_tp, factor] = find_tp(tp, MM, dG, Size)

kk = 1e6 / (dG*2.802e6);
NewSize  = kk / tp ;

if NewSize >= Size
  factor = NewSize / Size;
  new_tp = tp;
else
  % Planned matrix size
  MM2 = MM * NewSize/ Size;
  
  % Better matrix size
  MM3 = floor(MM2/2+0.5)*2;
  factor = MM3 / MM;
  
  new_tp = kk / (MM3 * Size / MM);
end
