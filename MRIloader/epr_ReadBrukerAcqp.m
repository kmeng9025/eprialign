function Parameters = epr_ReadBrukerAcqp(FileName)

Parameters = [];
fp = fopen(FileName, 'r');
if (fp == -1)
  error(['epr_ReadBrukerAcqp: couldn''t open parameter file "' FileName '".']);
end

% $$ index
idxSS = 1;

try
  root_parameter = [];
  array_idx = 0;  % amount of data field left
  array_dim = []; % dimension of data array
  while ~feof(fp)
    tmp = fgetl(fp);
    if ~isempty(strfind(tmp, '##$'))
      a = regexp(tmp, '##\$(?<field>\w+)=(?<value>.*)','names');
      root_parameter = a.field;
      Parameters.(root_parameter) = [];
      value = a.value;
      
      a = regexp(value, '\s*\(\s*(?<nval>\d+)\s*\)\s*','names');
      if ~isempty(a)
        % parameter, 1D
        array_idx = str2double(a.nval);
        array_dim = array_idx;
      else
        a = regexp(value, '\s*\(\s*(?<nval1>\d+)\s*,\s*(?<nval2>\d+)\s*\)\s*','names');
        if ~isempty(a)
          % parameter, 2D
          array_dim = [str2double(a.nval1), str2double(a.nval2)];
          array_idx = prod(array_dim);
        else
          a = regexp(value, '\s*\(\s*(?<nval1>\d+)\s*,\s*(?<nval2>\d+)\s*,\s*(?<nval3>\d+)\s*\)\s*','names');
          if ~isempty(a)
            % parameter, 3D
            array_dim = [str2double(a.nval1), str2double(a.nval2), str2double(a.nval3)];
            array_idx = prod(array_dim);
          else
            % assignement
            array_idx = 0;
            Parameters.(root_parameter)=strtrim(value);
            % convert to number
            val = str2double(Parameters.(root_parameter));
            if ~isnan(val), Parameters.(root_parameter) = val; end
          end
        end
      end
    elseif ~isempty(strfind(tmp, '##'))
      a = regexp(tmp, '##(?<field>\w+)=(?<value>.*)','names');
      Parameters.(a.field) = a.value;
    elseif ~isempty(strfind(tmp, '$$'))
      Parameters.SS{idxSS} = strtrim(tmp(strfind(tmp, '$$')+2:end));
      idxSS=idxSS+1;
    elseif array_idx >= 0
      if ~isempty(strfind(tmp, '<')) && ~isempty(strfind(tmp, '>'))
        % This is string
        tmp = tmp(tmp ~= '<' & tmp ~= '>');
        Parameters.(root_parameter) = tmp;
      else
        % this is an array
        a = str2num(tmp);
        na = length(a);
        if ~isnan(a)
          for jj=1:min(array_idx, na)
            Parameters.(root_parameter)(end+1)=a(jj);
          end
          array_idx  = array_idx - na;
          if array_idx <= 0
            try Parameters.(root_parameter) = reshape(Parameters.(root_parameter), array_dim);
            catch
            end;
          end
        end
      end
    else
      disp(['Root parameter ',root_parameter])
      disp(['Line ',tmp,'was not interpreted']);
    end
  end
catch er
  disp(['Root parameter ',root_parameter,', line: ',tmp])
  disp(er)
  error(['mrigMRI loader error: parsing parameter file "' FileName '".']);
end

fclose(fp);
