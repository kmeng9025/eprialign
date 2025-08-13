% EPRI_READIMAGEFILE - load image data
% [mat,mat_bl,out_parameters] = EPRI_READIMAGEFILE(file_name, load_opt);
% file_name    - Source data filename [string]
% load_opt - Loading options [structure]
%   [].to be done   - FBP structure parameters []
% mat               - Data [array nPoints x nTypes x nPrj]
% mat_bl            - Baseline for data [array nPoints x nTypes x nPrj]
% out_parameters    - Return parameter structure
%   [].raw_info     - Parameters of loaded projections [structure]
% See also EPRI_LOAD_FOR_PROCESSING, EPRI_ESE_PREPROCESS, EPRI_RECONSTRUCT.

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, 2014
% Contact: epri.uchicago.edu

function [out, raw_info, out_parameters]=epri_ReadImageFile(file_name, load_opt)

out_parameters = [];
ImagingModality = upper(safeget(load_opt, 'Modality', 'PULSEFBP'));
out_parameters.Modality = upper(ImagingModality);
out_parameters.Origin = upper(safeget(load_opt, 'Origin', 'SpecMan'));

[~, ~, fext]=fileparts(file_name);

BlFile = safeget(out_parameters, 'bl_file', []);

raw_info.ampHH = 1; % Hardware scaling factor

switch out_parameters.Modality
    case 'PULSEFBP'
        switch out_parameters.Origin
            case {'SPECMAN','SMTDMS'}
                switch upper(fext)
                    case '.D01',  [ax,data_and_baseline,dsc]=kv_d01read(file_name);
                    case '.TDMS',  [ax,data_and_baseline,dsc]=kv_smtdmsread(file_name);
                end
                
                % correction of SpecMan issue
                sz = size(data_and_baseline);
                if length(sz) > 2 && sz(3) < sz(2) && sz(3) ~= 1
                  data_and_baseline = permute(data_and_baseline,[1,3,2]);
                end
                % Get image description string
                image_description = safeget(dsc, 'exp_info_type1',[]);
                disp(['Image description: ',image_description]);
                
                % extract FBP protocol
                if ~isempty(image_description)
                    FBP = epri_DecodeImageType(image_description);
                    isDecoded = ~isempty(FBP);
                    isDecoded = isDecoded && isfield(FBP,'imtype');
                    if isDecoded, out_parameters.FBP = FBP;
                    else out_parameters.FBP = load_opt.FBP;
                    end
                end
                
                info = iradon_FBPGradTable(out_parameters.FBP);
                raw_info = CopyStructureFields(raw_info, info, {'G', 'gidx', 'nP', 'nTrace', 'Dim', ...
                    'deltaL', 'deltaH'}, '');
                raw_info.protocol = image_description;
                
                if strcmpi(safeget(out_parameters.FBP,'zerograd','none'),'EVERY_N'), % some kinda DAQ problem
                    if numel(info.gidx)<size(data_and_baseline,3),    % remove later
                        data_and_baseline = data_and_baseline(:,:,2:end);
                        ax.z = ax.z(1:end-1,:);
                    end
                end
                
                sz_data = size(data_and_baseline);
                trace_per_projection = prod(sz_data(2:end))/info.nTrace;
                time_trace_length = sz_data(1);
                
                if floor(trace_per_projection)*info.nTrace ~= prod(sz_data(2:end))
                    info
                    sz_data
                    error('epr_ReadPulseImageFile: data do not correspond to reading options.');
                end
                
                data_and_baseline = reshape(data_and_baseline, [sz_data(1)*trace_per_projection, info.nTrace]);
                sz_data = size(data_and_baseline);
                
                out_parameters.Sequence = load_opt.Sequence;
                switch out_parameters.Sequence
                    case {'2pECHO','2pECHOSRT'}
                        raw_info.tau2 = get_array(safeget(dsc, 'params_tau', '1 us'), length(ax.y))'*2;
                        raw_info.Trep = get_array(safeget(dsc, 'params_RepTime', '1 us'), length(ax.y))';
                        raw_info.t_ax = ax.x;
                    case '3pT1'
                        raw_info.tau2 = get_array(safeget(dsc, 'params_tau', '1 us'), length(ax.y))'*2;
                        raw_info.T1 = get_array(safeget(dsc, 'params_T1', '1 us'), length(ax.y))';
                        raw_info.Trep = get_array(safeget(dsc, 'params_RepTime', '1 us'), length(ax.y))';
                        raw_info.t_ax = ax.x;
                    case 'ESEInvRec'
                        raw_info.tau = get_array(safeget(dsc, 'params_tau', '1 us'), 1)';
                        raw_info.T1 = get_array(safeget(dsc, 'params_T1', '1 us'), length(ax.y))';
                        raw_info.Trep = get_array(safeget(dsc, 'params_RepTime', '1 us'), length(ax.y))';
                        raw_info.t_ax = ax.x;
                        if numel(raw_info.tau) == 1, raw_info.tau = raw_info.tau*ones(size(raw_info.T1)); end
                        raw_info.tau2 = 2* raw_info.tau;
                    case 'Rabi'
                        raw_info.tau = get_array(safeget(dsc, 'params_tau', '1 us'), 1)';
                        raw_info.tau2 = 2* raw_info.tau;
                        raw_info.tp = get_array(safeget(dsc, 'params_t90', '1 us'), length(ax.y))';
                        raw_info.t_ax = ax.x;
                end
                if isfield(dsc, 'params_Gx')
                    raw_info.Gexp(:,1) = get_array(load_long_field(dsc, 'params_Gx', ''), 10);
                    raw_info.Gexp(:,2) = get_array(load_long_field(dsc, 'params_Gy', ''), 10);
                    raw_info.Gexp(:,3) = get_array(load_long_field(dsc, 'params_Gz', ''), 10);
                end
                
                raw_info.Offset = get_array(load_long_field(dsc, 'params_Offset', '0 G'));
                
                if ~isempty(BlFile)
                    [ax1,bl]=kv_d01read(BlFile);
                    bl = reshape(bl, [sz_data(1)*trace_per_projection, 1]);
                    
                    data_and_baseline(:,end+1) = bl;
                    info.BL(end+1) = 1;
                end
                
                % timestamp
                raw_info.StartTime = ax.StartTime;
                raw_info.FinishTime = ax.FinishTime;
                raw_info.ExpTime = ax.ExpTime;
                
                % calculated imaging time
                reprate = ax.RepTime;
                if isfield(dsc, 'params_XReps')
                    reps = sum(get_array(dsc.params_XReps));
                    prj   = get_size(dsc.sweep_sweep2);
                else
                    reps = ax.shots;
                    prj   = length(ax.y);
                    if isfield(ax, 'z'), prj = prj * length(ax.z); end
                end
                raw_info.CalculatedTime = reprate*reps*prj/60.;
            otherwise
                error('ReadPulseImageFile: Unknown data origin.');
        end
        
        switch safeget(out_parameters.FBP, 'scheme', 'single_b')
            case 'single_b'
                processed_info = raw_info;
                [data_and_baseline, out.zero_g, service_idx, processed_info] = epri_zerog_split(data_and_baseline, info.service_idx, processed_info);

                [out.mat, out.mat_bl, service_idx, processed_info] = epri_baseline_split(data_and_baseline, service_idx, processed_info);
                
                [out.mat, out.mat_nav] = epri_navigator_split(out.mat, service_idx, processed_info);
                if ~isempty(out.mat_bl)
                    [out.mat_bl, out.mat_bl_nav, service_idx, nav_service_idx, processed_info, nav_processed_info] = ...
                        epri_navigator_split(out.mat_bl, service_idx, processed_info);
                    out = CopyStructureFields(out, nav_processed_info, {'G', 'Gexp'}, 'nav_');
                end
                out = CopyStructureFields(out, processed_info, {'G', 'Gexp'}, '');
                
                info.nN = size(out.mat_nav, 2);
                info.nG = size(out.zero_g, 2);
                
                % reshape data and navigators
                if ~isempty(out.zero_g)
                    out.zero_g = reshape(out.zero_g, [time_trace_length,trace_per_projection, info.nG]);
                end
                
                out.mat = reshape(out.mat, [time_trace_length,trace_per_projection, info.nP]);
                if ~isempty(out.mat_bl)
                    out.mat_bl = reshape(out.mat_bl, [time_trace_length,trace_per_projection, info.nP]);
                end
                
                out.mat_nav = reshape(out.mat_nav, [time_trace_length,trace_per_projection, info.nN]);
                if ~isempty(out.mat_bl)
                    out.mat_bl_nav = reshape(out.mat_bl_nav, [time_trace_length,trace_per_projection, info.nN]);
                end
                
                raw_info.data = out_parameters;
                raw_info.nP  = info.nP;
                raw_info.Dim = info.Dim;
                raw_info.ax = ax;
                raw_info.Unformatted = dsc;
            case 'multi_b'
                raw_info.data = out_parameters;
                raw_info.nP  = info.nP;
                raw_info.Dim = info.Dim;
                NOTBLidx = info.gidx>0;
                BLidx = info.gidx<=0;
                raw_info.GradX = info.GradX(NOTBLidx);
                raw_info.GradY = info.GradY(NOTBLidx);
                raw_info.GradZ = info.GradZ(NOTBLidx);
                raw_info.gidx  = info.gidx(NOTBLidx); % for the time being... sort of cheating);
                raw_info.Boffset = info.Boffset(NOTBLidx);
                raw_info.BLoffset = info.BLoffset;
                % Here the data structure is:
                % size(mat) = [time_trace_length; number_of_different_tau; number_of_projections]
                
                raw_info.ax = ax;
                raw_info.Unformatted = dsc;
                nSubProj = length(raw_info.gidx);
                
                % construct data
                mat    = reshape(data_and_baseline(:, NOTBLidx), time_trace_length,trace_per_projection, nSubProj);
                
                % construct baseline
                sz_mat_bl =[time_trace_length,trace_per_projection, nSubProj];
                bl_idx = find(BLidx);
                
                if isempty(bl_idx)
                    mat_bl = zeros(sz_mat_bl);
                    raw_info.BL = [];
                elseif   length(bl_idx) == 1
                    raw_info.BL = zeros(prod(sz_mat(2:end)),1);
                    raw_info.BL(1) = 1;
                    mat_bl = data_and_baseline(:, bl_idx(1)*ones(prod(sz_mat(2:end)),1));
                else
                    mat_bl = data_and_baseline(:,bl_idx);
                    mat_bl = reshape(mat_bl, [time_trace_length,trace_per_projection,length(bl_idx)]);
                    
                    % base line is acquired PRIOR to first trace
                    bl_shift = bl_idx - (0:(length(bl_idx)-1))';
                    % last base line is acquired post last trace
                    bl_shift(end) =  bl_shift(end) - 1;
                    raw_info.BL = bl_shift;
                    
                    if length(bl_shift) ~= nSubProj
                        mat_bl_out = zeros(sz_mat_bl);
                        for ii=1:trace_per_projection
                            % interpolate baseline
                            for jj=1:size(mat, 1)
                                mat_bl_out(jj,ii,:) = interp1(bl_shift, squeeze(mat_bl(jj,ii,:)), 1:nSubProj,'spline');
                            end
                        end
                        mat_bl = mat_bl_out;
                    end
                end
        end
    case 'PULSESPI'
        switch out_parameters.Origin
            case 'SPECMAN'
                switch upper(fext)
                    case '.D01',  [ax,data_and_baseline,dsc]=kv_d01read(file_name);
                    case '.TDMS',  [ax,data_and_baseline,dsc]=kv_smtdmsread(file_name);
                end
                % Get image description string
                image_description = safeget(dsc, 'exp_info_type1',[]);
                disp(['Image description: ',image_description]);
                
                out_parameters.Sequence = load_opt.Sequence;
                if ~isempty(image_description)
                    SPI = epri_DecodeImageType(image_description);
                    if ~isempty(SPI), out_parameters.SPI = SPI;
                    else out_parameters.SPI = load_opt.SPI;
                    end
                end
                
                if isfield(dsc, 'params_Gx')
                    nGradients = size(data_and_baseline, 2);
                    Gx = get_array(load_long_field(dsc, 'params_Gx', ''), 10);
                    Gy = get_array(load_long_field(dsc, 'params_Gy', ''), 10);
                    Gz = get_array(load_long_field(dsc, 'params_Gz', ''), 10);
                    Offset = get_array(load_long_field(dsc, 'params_Offset', ''), 10);
                  
                    raw_info.Gexp(:,1) = Gx(1:nGradients);
                    raw_info.Gexp(:,2) = Gy(1:nGradients);
                    raw_info.Gexp(:,3) = Gz(1:nGradients);
                    raw_info.Offset = Offset(1:nGradients);
                end
                
                rebuild_index = true;
                
                if rebuild_index
                  info1.service_idx = abs(raw_info.Offset) < 0.1;
                  info1.G = raw_info.Gexp;
                  info1.nP = numel(find(info1.service_idx == 1));
                  Gmax   = out_parameters.SPI.MaxGradient;
                  Gexp = raw_info.Gexp(info1.service_idx, :);
                  info1.pidx.i = 1+floor((Gexp(:,1)+Gmax) * (out_parameters.SPI.nSteps(1)-1)/ 2 / Gmax+0.5);
                  info1.pidx.j = 1+floor((Gexp(:,2)+Gmax) * (out_parameters.SPI.nSteps(2)-1)/ 2 / Gmax+0.5);
                  info1.pidx.k = 1+floor((Gexp(:,3)+Gmax) * (out_parameters.SPI.nSteps(3)-1)/ 2 / Gmax+0.5);
                  info1.gidx = sub2ind(out_parameters.SPI.nSteps, info1.pidx.i, info1.pidx.j, info1.pidx.k);
                  info1.Dim = out_parameters.SPI.nSteps;
                  info1.nTrace = length(info1.service_idx);
                  
                  info = info1;
                  
                else
                  info = td_GetSPIGradientTable(out_parameters.SPI);
                end
                raw_info = CopyStructureFields(raw_info, info, {'G', 'gidx', 'nP', 'nTrace', 'Dim'}, '');
                processed_info = raw_info;
                
                sz_data = size(data_and_baseline);
                if length(sz_data) == 2
                  data_and_baseline = reshape(data_and_baseline, [sz_data(1),1,sz_data(2)]);
                  sz_data = size(data_and_baseline);
                end
                
                time_trace_length = sz_data(1);
                trace_per_projection = prod(sz_data(2:end))/info.nTrace;
                if fix(trace_per_projection)*info.nTrace ~= prod(sz_data(2:end))
                  info
                  sz_data
                  error('ReadPulseImageFile: data do not correspond to reading options.');
                end
                
                [out.mat, out.mat_bl, service_idx, processed_info, bl_info] = epri_baseline_split(...
                  reshape(data_and_baseline, [sz_data(1)*sz_data(2), sz_data(3)]), info.service_idx, processed_info);
                
                raw_info.data = out_parameters;
                
                % timestamp
                raw_info.StartTime = ax.StartTime;
                raw_info.FinishTime = ax.FinishTime;
                raw_info.ExpTime = ax.ExpTime;
                raw_info.MaxGradient = out_parameters.SPI.MaxGradient;
                
                raw_info.t_ax = ax.x;
                
                switch out_parameters.Sequence
                    case 'FIDInvRec'
                        raw_info.tau = get_array(safeget(dsc, 'params_tau', '1 us'), 1)';
                        raw_info.T = get_array(safeget(dsc, 'params_T', '1 us'), length(ax.y))';
                        raw_info.Trep = get_array(safeget(dsc, 'params_RepTime', '1 us'), length(ax.y))';
                        raw_info.t_ax = ax.x;
                    case {'FIDSRT'}
                        raw_info.tau2 = get_array(safeget(dsc, 'params_tau', '1 us'), length(ax.y))'*2;
                        raw_info.Trep = get_array(safeget(dsc, 'params_RepTime', '1 us'), length(ax.y))';
                        raw_info.t_ax = ax.x;
                end
                
                out.mat = reshape(out.mat, [time_trace_length, sz_data(2), raw_info.nP]);
                out.mat_bl = reshape(out.mat_bl, [time_trace_length, sz_data(2), raw_info.nP]);
                out.raw_info = raw_info;
                
            otherwise
                error('ReadPulseImageFile: Unknown data origin.');
        end
        
        raw_info.ax = ax;
        raw_info.dsc = dsc;
        
    case 'RSFBP'
        switch out_parameters.Origin
            case 'SPECMAN'
                [ax,out.mat,dsc]=kv_d01read(file_name);
                % wrong triad for Re and Im channels
                out.mat = real(out.mat)-1i*imag(out.mat);
                out.mat_bl = [];
                
                % Get image description string
                image_description = safeget(dsc, 'exp_info_type1',[]);
                disp(['Image description: ',image_description]);
                % extract FBP protocol
                if ~isempty(image_description),
                    FBP = epri_DecodeImageType(image_description);
                    isDecoded = ~isempty(FBP);
                    isDecoded = isDecoded && isfield(FBP,'imtype');
                    if isDecoded, out_parameters.FBP = FBP;
                    else out_parameters.FBP = load_opt.FBP;
                    end
                end
                
                info = iradon_FBPGradTable(out_parameters.FBP);
                raw_info = CopyStructureFields(raw_info, info, {'GradX', 'GradY', 'GradZ', 'gidx', 'nP', 'nTrace', 'Dim', ...
                    'deltaL', 'deltaH', 'service_idx'}, '');
                raw_info.protocol = image_description;
                
                if isfield(dsc, 'params_Gx')
                    raw_info.Gexp(:,1) = get_array(load_long_field(dsc, 'params_Gx', ''), 10);
                    raw_info.Gexp(:,2) = get_array(load_long_field(dsc, 'params_Gy', ''), 10);
                    raw_info.Gexp(:,3) = get_array(load_long_field(dsc, 'params_Gz', ''), 10);
                end
                
                sz_data = size(out.mat);
                if raw_info.nTrace ~= prod(sz_data(2:end))
                    disp('Parameters given:'); disp(info)
                    disp('Dimensions read'); disp(sz_data)
                    error('ReadPulseImageFile: data do not correspond to reading options.');
                end
                
                raw_info.data = out_parameters;
                % timestamp
                raw_info.StartTime = ax.StartTime;
                raw_info.FinishTime = ax.FinishTime;
                raw_info.ExpTime = ax.ExpTime;
                raw_info.sampling = get_array(safeget(dsc, 'aliases_RSdt', '20 ns'))';
                
                if isfield(dsc, 'params_ScanWidth')
                    raw_info.FieldSweep = get_array(load_long_field(dsc, 'params_ScanWidth', '10 G'))';
                else
                    raw_info.FieldSweep = get_array(load_long_field(dsc, 'aliases_RSwidth', '10 G'))';
                end
                raw_info.RSfrequency = get_array(load_long_field(dsc, 'aliases_RSfrequency', '4 kHz'))';
                
                if length(raw_info.RSfrequency) == 1, raw_info.RSfrequency = raw_info.RSfrequency * ones(raw_info.nTrace,1); end
                if length(raw_info.sampling) == 1, raw_info.sampling = raw_info.sampling * ones(raw_info.nTrace,1); end
                if length(raw_info.FieldSweep) == 1, raw_info.FieldSweep = raw_info.FieldSweep * ones(raw_info.nTrace,1); end
                raw_info.Unformatted = dsc;
          case 'BRUKER'
            [t,out.mat]=eprload(file_name);
            n_prj = size(out.mat, 2);
            raw_info.sampling = mean(diff(t{1}));
            raw_info.data.FBP = [];
            raw_info.RSfrequency = safeget(load_opt.FBP, 'RSfrequency', 5.02e3)*ones(n_prj,1); % Hz
            raw_info.FieldSweep = safeget(load_opt.FBP, 'RSsweep', 31.557)*ones(n_prj,1); % gauss
            raw_info.sampling = min(diff(t{1}))*1e-3*ones(n_prj,1);
            out.mat_bl = [];
        end
  case 'CWFBP'
    [ax,out.mat,dsc]=brukerread(file_name);
    out.mat_bl = [];
    raw_info.FieldMin = repmat(min(ax.x), size(out.mat, 2), 1);
        raw_info.FieldMax = repmat(max(ax.x), size(out.mat, 2), 1);
        raw_info.CF = ax.cf * 1E4;
        raw_info.data.FBP = load_opt.FBP;
end

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

% --------------------------------------------------------------------
function [val] = get_array(str, n)
[str] = strtok(str,';');
if contains(str, 'logto')
    a = regexp(str, '\s*(?<data1>[0-9.eE-+]+[\sa-z_A-Z]*\w*)\s*logto\s*(?<data2>[0-9.eE-+]+[\sa-z_A-Z]*\w*)','names');
    if isfield(a, 'data1') && isfield(a, 'data2')
        val = logspace(log10(get_val(a.data1)), log10(get_val(a.data2)), n);
    end
elseif contains(str, 'to')
    a = regexp(str, '\s*(?<data1>[0-9.eE-+]+[\sa-z_A-Z]*\w*)\s*to\s*(?<data2>[0-9.eE-+]+[\sa-z_A-Z]*\w*)','names');
    if isfield(a, 'data1') && isfield(a, 'data2')
        val = linspace(get_val(a.data1), get_val(a.data2), n);
    end
elseif contains(str, 'step')
    a = regexp(str, '\s*(?<data1>[0-9.eE-+]+[\sa-z_A-Z]*\w*)\s*step\s*(?<data2>[0-9.eE-+]+[\sa-z_A-Z]*\w*)','names');
    if isfield(a, 'data1') && isfield(a, 'data2')
        val = get_val(a.data1)+(0:n-1)*get_val(a.data2);
    end
elseif contains(str, ':')
    pos = strfind(str, ':');
    val = sscanf(str(pos+1:end), '%g,');
    theunit = str(1:pos);
    if contains(theunit, 'mG/cm')
      val = val * 1e-3;
    end
else
    a = regexp(str, '\s*(?<data>[0-9.eE\-\+]+[\sa-z_A-Z]*\w*),*','names');
    val = [];
    for ii=1:length(a); val(end+1) = str2val(a(ii).data); end
end
val = val(:);

% --------------------------------------------------------------------
function full_str = load_long_field(dsc, the_field, the_default)

str = safeget(dsc, the_field, the_default);

if contains(str, '***')
    full_str = '';
    for ii=0:10000
        str = safeget(dsc, sprintf('%s_%i',the_field,ii), '***');
        if contains(str, '***'), break; end;
        str = strtrim(str);
        full_str = [full_str, str(str~='*')];
    end
else
    full_str = str;
end

% --------------------------------------------------------------------
function val = get_size(str)

[str, str1] = strtok(str,',');
str = strtok(str1(2:end),',');
val = str2num(str);

% --------------------------------------------------------------------
function val = get_repetitions(str)

[str, str1] = strtok(str,',');
str = strtok(str1(2:end),',');
[str, str1] = strtok(str1(2:end),',');
str = strtok(str1(2:end),',');
val = str2num(str);

% --------------------------------------------------------------------
function st_dest = CopyStructureFields(st_dest, st_source, fields, prefix)
for ii=1:length(fields)
    st_dest.([prefix, fields{ii}]) = safeget(st_source, fields{ii}, []);
end