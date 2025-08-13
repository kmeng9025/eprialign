% AXstructure = kv3_ParameterReader('load', ax, dsc)
% ListOfOptions = kv3_ParameterReader('options')
function ax = kv3_ParameterReader(mode, ax, dsc)

% reads fields
% freq1
% cf
% sample
% comment

switch mode
  case 'options'
    ax = {};
    ax{end+1} = struct('title', 'title', 'prefix', 'data:', 'value', '', 'type', 's', 'about', 'Title');
    ax{end+1} = struct('title', 'sample', 'prefix', 'data:', 'value', '', 'type', 's', 'about', 'Sample');
    ax{end+1} = struct('title', 'comment', 'prefix', 'data:', 'value', '', 'type', 's', 'about', 'Comment');
    ax{end+1} = struct('title', 'freq1', 'prefix', 'data:', 'value', 0.0, 'type', 'f', 'about', 'EPR frequency');
    ax{end+1} = struct('title', 'reference', 'prefix', 'data:', 'value', 0.0, 'type', 'f', 'about', 'Reference frequency');
    ax{end+1} = struct('title', 'cf', 'prefix', 'data:', 'value', 0.0, 'type', 'f', 'about', 'Central Field');
    ax{end+1} = struct('title', 'unit', 'prefix', 'data:', 'value', '', 'type', 's', 'about', 'Unit');
  case 'load'
    % determine unit
    switch safeget(ax, 'xlabel', '')
      case 'Field, G', ax.unit = 'G';
      case 'MagneticField, G', ax.unit = 'G';
      case 'Time, s', ax.unit = 's';
      case 'a, s', ax.unit = 's';
      case 't90, s', ax.unit = 's';
      case 'RF1, MHz', ax.unit = 'Hz';
    end
    
    switch safeget(dsc, 'KAZANformat', '')
      
      case 'SPECMANTDMS'
        ax.title = safeget(dsc, 'name', '');
        device = safeget(dsc, 'devices_BRIDGE', []);
        ax.freq1 = kvgetvalue(safeget(device, 'Frequency', '0.0'));
        sample_info = safeget(dsc, 'sample_info', '');
        ax.sample = safeget(sample_info, 'sample_info', '');
        
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
      otherwise
    end
end

function varargout = stripunit(str)
varargout{1} = str((str>='0' & str<='9') | str=='.' | str=='E');