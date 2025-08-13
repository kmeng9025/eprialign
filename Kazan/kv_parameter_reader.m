function ax = kv_parameter_reader(ax, dsc)

% reads fields
% freq1
% cf
% sample
% comment

switch safeget(dsc, 'KAZANformat', '')

  case 'SPECMAND01'
    if isfield(dsc,'general_freq1')
      ax.freq1 = kvgetvalue(dsc.general_freq1);
    end
    ax.title = safeget(dsc, 'general_name', '');
    ax.sample = safeget(dsc, 'sample_info_sample_info_0', '');
    ax.comment = safeget(dsc, 'sample_info_sample_info_1', '');
  case 'SPECMANNISTD01'  
    if isfield(dsc,'general_freq1')
      ax.freq1 = kvgetvalue(dsc.general_freq1);
    end
    
  case 'BRUKERDSC'
    % extracting some useful parameters from file
    if isfield(dsc, 'MWFQ')
        ax.freq1 = str2double(dsc.MWFQ);
    end
    if isfield(dsc, 'MF')
        ax.freq1 = str2double(dsc.MF);
    end
    if isfield(dsc, 'CenterField')
        ax.cf = str2double(stripunit(dsc.CenterField))*1E-4;
    end
    if isfield(dsc, 'TITL')
        ax.title = dsc.TITL;
    elseif isfield(dsc, 'JCO')
        ax.title = dsc.JCO;
    else
        ax.title = '?';
    end
    ax.sample = safeget(dsc, 'SAMP', ''); 
    ax.comment = safeget(dsc, 'CMNT', ''); 
end

function varargout = stripunit(str)
varargout{1} = str((str>='0' & str<='9') | str=='.' | str=='E');