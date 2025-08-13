% FinalImage = ese_fbp(file_name, file_suffix, output_path, fields)
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

function FinalImage = spec_grad_min(file_name, file_suffix, output_path, fields)

is_fit_data = strcmp(safeget(fields.prc, 'fit_data','yes'),'yes');
is_recon_data = strcmp(safeget(fields.prc, 'recon_data','yes'),'yes');
is_export_proj = strcmp(safeget(fields.prc, 'export_prj','yes'),'yes');
is_save_data = strcmp(safeget(fields.prc, 'save_data','yes'),'yes');

fnames = epri_filename(file_name, file_suffix, output_path);

load_opt.Modality = 'PULSEFBP';
load_opt.Sequence  = '2pECHO';
load_opt.FBP = fields.fbp;

[fp, fn, fext]=fileparts(file_name);
switch upper(fext)
    case '.D01',  [ax,data_and_baseline,dsc]=kv_d01read(file_name);
    case '.TDMS',  [ax,data_and_baseline,dsc]=kv_smtdmsread(file_name);
end

% extractposition information
Xstr = load_long_field(dsc, 'params_Gx', '');
Ystr = load_long_field(dsc, 'params_Gy', '');
Zstr = load_long_field(dsc, 'params_Gz', '');
% Offstr =  load_long_field(dsc, 'params_Offset', '');

X = get_array(Xstr, length(ax.y)); % mm
Y = get_array(Ystr, length(ax.y)); % mm
Z = get_array(Zstr, length(ax.y)); % mm
% OFF = get_array(Offstr); % B offset

% if length(X) ~= length(Y) || length(X) ~= length(Z)
%     error('Wrong image axis sizes');
% end
% 
% if length(OFF) ~= length(X)
%     OFF = zeros(size(X));
% end

% remove baseline
% bl_idx = abs(OFF) > 5; 
% X = X(~bl_idx);
% Y = Y(~bl_idx);
% Z = Z(~bl_idx);
data_only = data_and_baseline;
% bl_only = data_and_baseline(:, bl_idx);

% if ~isempty(find(bl_idx, 1))
%     if   length(find(bl_idx)) == 1
%         baseline = repmat(bl_only, size(data, 2));
%     else
%         baseline = zeros(size(data_only));
%         pBLidx = find(bl_idx);
%         pNOTBLidx = find(~bl_idx);
%         for jj=1:size(data_only, 1)
%             baseline(jj,:) = interp1(pBLidx, bl_only(jj,:), pNOTBLidx,'spline');
%         end
%     end
%     data_only = data_only - baseline;
% end

% determine cube of data
% xx = unique(X);
% xy = unique(Y);
% xz = unique(Z);
% 
% cx = mean(xx);
% cy = mean(xy);
% cz = mean(xz);

% fft
fields.fft.fft = 1;
fields.fft.data = '0_';
fields.fft.lshift = 120;
fields.fft.opt='complex';
fields.fft.zerofill = 8;

data_only = squeeze(kv_baseline_correction(data_only, fields.td));
sz = size(data_only);
[rax, ry] = kv_fft(ax, reshape(data_only, [sz(1),prod(sz(2:end))]), fields.fft);
szy = size(ry);

amp = max(abs(ry), [], 1);
amp = reshape(amp, sz(2:end));
% MatrixGUI(amp)

FinalImage.AMP = amp;
% FinalImage.Raw = reshape(ry, [szy(1),sz(2:end)]);

% --------------------------------------------------------------------
function s = copy_fields(s, s1)

fname = fieldnames(s1);
for ii=1:length(fname), s.(fname{ii}) = s1.(fname{ii}); end

% --------------------------------------------------------------------
function full_str = load_long_field(dsc, the_field, the_default)

str = safeget(dsc, the_field, the_default);

if contains(str, '***')
    full_str = '';
    for ii=0:10000
        str = safeget(dsc, sprintf('%s_%i',the_field,ii), '***');
        if contains(str, '***'), break; end
        str = strtrim(str);
        full_str = [full_str, str(str~='*')];
    end
else
    full_str = str;
end

% --------------------------------------------------------------------
function [val] = get_array(str, n)
[str] = strtok(str,';');
if contains(str, 'logto')
    a = regexp(str, '\s*(?<data1>[0-9.eE-+]+[\sa-z_A-Z]*\w*)\s*logto\s*(?<data2>[0-9.eE-+]+[\sa-z_A-Z]*\w*)','names');
    if isfield(a, 'data1') && isfield(a, 'data2')
        val = logspace(log10(get_val(a.data1)), log10(get_val(a.data2)), n);
    end
elseif contains(str, 'to')
    a = regexp(str, '\s*(?<data1>[0-9.eE-+]+\s*[a-z_A-Z/]*\w*)\s*to\s*(?<data2>[0-9.eE-+]+[\sa-z_A-Z]*\w*)','names');
    if isfield(a, 'data1') && isfield(a, 'data2')
        val = linspace(get_val(a.data1), get_val(a.data2), n);
    end
elseif contains(str, 'step')
    a = regexp(str, '\s*(?<data1>[0-9.eE-+]+[\sa-z_A-Z]*\w*)\s*step\s*(?<data2>[0-9.eE-+]+[\sa-z_A-Z]*\w*)','names');
    if isfield(a, 'data1') && isfield(a, 'data2')
        val = get_val(a.data1)+(0:n-1)*get_val(a.data2);
    end
else
    a = regexp(str, '\s*(?<data>[0-9.eE\-\+]+[\sa-z_A-Z]*\w*),*','names');
    val = [];
    for ii=1:length(a); val(end+1) = str2val(a(ii).data); end
end
val = val(:);

% --------------------------------------------------------------------
function [val, str_unit, str_koefficient] = get_val(str)

[str1] = strtok(str,';');
[val, str_unit, str_koefficient] = str2val(str1);

% --------------------------------------------------------------------
function [val, unit, pref, pref_val] = str2val(str)
prefix = ['p','n', 'u', 'm', 'k', 'M', 'G', 'T'];
koeff  = [1E-12, 1E-9, 1E-6, 1E-3, 1E3, 1E6, 1E9, 1E12];
pref = ''; pref_val = 1;

res = regexp(str, '(?<number>[0-9.eE-+]+)\s*(?<unit>\w+)*', 'names');

if ~isempty(res)
    res = res(1);
    val = str2double(res.number);
    if isfield(res, 'number'), unit = res.unit; else unit = ''; end
    if length(unit) > 1
        if ~isempty(unit)
            kk = findstr(prefix, unit(1));
            if ~isempty(kk)
                val = val * koeff(kk);
                unit = unit(2:end);
                pref = prefix(kk);
                pref_val = koeff(kk);
            end
        end
    end
end
