function [ss_out, ss1_out] = ProcessFormController(ss_in_ini, ss_in_fields, fields_data, action)
% ProcessFormController(ss, fields_data, 'fields2gui');
% [ini_fields, ss] = ProcessFormController(ss, fields_data, 'gui2ini');
% ss = ProcessFormController(ss, fields_data, 'ini2gui');

if iscell(fields_data)
  switch upper(action)
    case 'INI2FIELDS'
      ss_out  = ss_in_fields;
      ss1_out = ss_in_ini;
      for ii=1:length(fields_data)
        [ss_out, ss1_out] = ProcessFormController(ss1_out, ss_out, fields_data{ii}, 'INI2FIELDS');
      end
    case 'INI2GUI'
      ss_out  = ss_in_fields;
      ss1_out = ss_in_ini;
      for ii=1:length(fields_data)
        ss_out = ProcessFormController(ss_in_ini, ss_out, fields_data{ii}, 'INI2GUI');
      end
    case 'FIELDS2INI'
      ss_out = ss_in_ini;
      ss1_out = ss_in_fields;
      for ii=1:length(fields_data)
        [ss_out, ss1_out] = ProcessFormController(ss_out, ss1_out, fields_data{ii}, 'FIELDS2INI');
      end
    case 'FIELDS2GUI'
      ss_out  = ss_in_fields;
      ss1_out = ss_in_ini;
      for ii=1:length(fields_data)
        [ss_out, ss1_out] = ProcessFormController(ss1_out, ss_out, fields_data{ii}, 'FIELDS2GUI');
      end
    case 'GUI2FIELDS'
      ss_out  = ss_in_fields;
      ss1_out = ss_in_ini;
      for ii=1:length(fields_data)
        [ss_out, ss1_out] = ProcessFormController(ss1_out, ss_out, fields_data{ii}, 'GUI2FIELDS');
      end
    case 'GUI2INI'
      ss_out = ss_in_ini;
      ss1_out = ss_in_fields;
      for ii=1:length(fields_data)
        [ss_out, ss1_out] = ProcessFormController(ss_out, ss1_out, fields_data{ii}, 'GUI2INI');
      end
  end
else
  try
    switch upper(action)
      case 'INI2FIELDS'
        field2string = @(data, fld) safeget(data, fld.Field, fld.Value);
        ss_out = ss_in_fields;
        switch fields_data.Type
          case {'S','F'}
            ss_out.(fields_data.Field)= field2string(ss_in_ini, fields_data);
          case 'D'
            ss_out.(fields_data.Field)= field2double(ss_in_ini, fields_data);
          case 'bool'
            ss_out.(fields_data.Field)= field2double(ss_in_ini, fields_data);
          case {'IDX', 'IDX0', 'IDX1'}
            ss_out.(fields_data.Field)= field2double(ss_in_ini, fields_data);
          case {'IDX1+'}
            ss_out.(fields_data.Field)= field2double(ss_in_ini, fields_data);
          case 'IDXS'
            val_string = field2string(ss_in_ini, fields_data);
            idx = -1;
            for jj=1:length(fields_data.Options)
              if strcmp(fields_data.Options{jj}, val_string), idx = jj; break; end
            end
            ss_out.(fields_data.Field) = fields_data.Options{jj};
        end
        if ~isfield(ss_in_ini, fields_data.Field), ss_in_ini = ProcessFormController(ss_in_ini, ss_out, fields_data, 'fields2ini'); end
        ss1_out = ss_in_ini;
      case 'FIELDS2INI'
        ss_out = ss_in_ini; ss1_out = ss_in_fields;
        switch fields_data.Type
          case {'S','F'}
            ss_out.(fields_data.Field)= ss_in_fields.(fields_data.Field);
          case 'D'
            if length(ss_in_fields.(fields_data.Field)) == 1
              ss_out.(fields_data.Field)= num2str(ss_in_fields.(fields_data.Field));
            elseif isempty(ss_in_fields.(fields_data.Field))
              ss_out.(fields_data.Field)= '[]';
            else
              val = ss_in_fields.(fields_data.Field);
              str = num2str(val(1));
              for jj=1:length(val)
              end
            end
          case 'bool'
            ss_out.(fields_data.Field)= num2str(iff(ss_in_fields.(fields_data.Field), true, false));
          case {'IDX','IDX0','IDX1', 'IDX1+'}
            ss_out.(fields_data.Field)= num2str(ss_in_fields.(fields_data.Field));
          case 'IDXS'
            ss_out.(fields_data.Field)= ss_in_fields.(fields_data.Field);
        end
      case 'FIELDS2GUI'
        ss_out  = ss_in_fields;
        ss1_out = ss_in_ini;
        switch fields_data.Type
          case {'S','F'}
            set(fields_data.Control, 'String', ss_in_fields.(fields_data.Field));
          case 'D'
            if numel(ss_in_fields.(fields_data.Field)) == 1
              set(fields_data.Control, 'String', num2str(ss_in_fields.(fields_data.Field)));
            end
          case 'bool'
            set(fields_data.Control, 'Value', ss_in_fields.(fields_data.Field));
          case 'IDX1+'
            set(fields_data.Control, 'Value', ss_in_fields.(fields_data.Field)+1);
          case 'IDXS'
            val_string = ss_in_fields.(fields_data.Field);
            idx = -1;
            for jj=1:length(fields_data.Options)
              if strcmp(fields_data.Options{jj}, val_string), idx = jj; break; end
            end
            set(fields_data.Control, 'Value', idx);
        end
      case 'GUI2FIELDS'
        ss_out = ss_in_fields; ss1_out = [];
        switch fields_data.Type
          case {'S','F'}
            ss_out.(fields_data.Field) = get(fields_data.Control, 'String');
          case 'D'
            ss_out.(fields_data.Field) = str2num(get(fields_data.Control, 'String'));
          case 'bool'
            ss_out.(fields_data.Field) = get(fields_data.Control, 'Value');
          case 'IDX1+'
            ss_out.(fields_data.Field) = get(fields_data.Control, 'Value') - 1;
          case 'IDXS'
            idx = get(fields_data.Control, 'Value');
            ss_out.(fields_data.Field) = fields_data.Options{idx};
        end
      case 'GUI2INI'
        ss1_out = ss_in_fields; ss_out = ss_in_ini;
        switch fields_data.Type
          case 'D'
            ss_out.(fields_data.Field)  = strtrim(get(fields_data.Control, 'String'));
            if isempty(ss_out.(fields_data.Field)),
              ss1_out.(fields_data.Field) = [];
            else
              try ss1_out.(fields_data.Field) = eval(ss_out.(fields_data.Field));
              catch err
                disp(err)
              end
            end
          case {'S','F'}
            ss_out.(fields_data.Field)  = strtrim(get(fields_data.Control, 'String'));
            ss1_out.(fields_data.Field) = ss_out.(fields_data.Field);
          case 'bool'
            ss1_out.(fields_data.Field) = get(fields_data.Control, 'Value');
            ss_out.(fields_data.Field) = num2str(iff(ss1_out.(fields_data.Field), true, false));
          case 'IDX'
            ss1_out.(fields_data.Field) = get(fields_data.Control, 'Value');
            ss_out.(fields_data.Field) = num2str(ss1_out.(fields_data.Field));
          case 'IDX1+'
            ss1_out.(fields_data.Field) = get(fields_data.Control, 'Value') - 1;
            ss_out.(fields_data.Field) = num2str(ss1_out.(fields_data.Field));
          case 'IDXS'
            idx = get(fields_data.Control, 'Value');
            ss_out.(fields_data.Field) = fields_data.Options{idx};
            ss1_out.(fields_data.Field) = fields_data.Options{idx};
        end
      case 'INI2GUI'
        field2string = @(data, fld) safeget(data, fld.Field, fld.Value);
        ss_out  = ss_in_fields;
        ss1_out = ss_in_ini;
        switch fields_data.Type
          case 'D'
            set(fields_data.Control, 'String', field2string(ss_in_ini, fields_data));
            ss_out.(fields_data.Field)= field2double(ss1_out, fields_data);
          case {'S','F'}
            set(fields_data.Control, 'String', field2string(ss_in_ini, fields_data));
            ss_out.(fields_data.Field)= field2string(ss1_out, fields_data);
          case 'bool'
            tmp = field2string(ss_in_ini, fields_data);
            if length(tmp) == 1, tmp = str2double(tmp);
            elseif strcmpi(tmp, 'YES'), tmp = 1;
            else tmp = 0;
            end
            switch get(fields_data.Control, 'Style')
              case 'checkbox', set(fields_data.Control, 'Value', tmp);
              case 'popupmenu', set(fields_data.Control, 'Value', tmp);
            end
            ss_out.(fields_data.Field)= tmp;
          case 'IDX'
            ss_out.(fields_data.Field)= field2double(ss_in_ini, fields_data);
            set(fields_data.Control, 'Value', ss_out.(fields_data.Field));
          case 'IDX1+'
            ss_out.(fields_data.Field)= field2double(ss_in_ini, fields_data);
            set(fields_data.Control, 'Value', ss_out.(fields_data.Field)+1);
          case 'IDXS'
            val_string = field2string(ss_in_ini, fields_data);
            idx = -1;
            for jj=1:length(fields_data.Options)
              if strcmp(fields_data.Options{jj}, val_string), idx = jj; break; end
            end
            if idx > length(fields_data.Options) || idx < 0, idx = 1; end;
            set(fields_data.Control, 'Value', idx);
            ss_out.(fields_data.Field)=fields_data.Options{idx};
        end
    end
  catch err
    disp(sprintf('Error %s.%s: %s', safeget(fields_data, 'Group', ''), fields_data.Field, err.message))
    ss_out = []; ss1_out = [];
  end
end

% --------------------------------------------------------------------
function  res = field2double(data, fld)

val = safeget(data, fld.Field, num2str(fld.Value));
res = str2num(val);
if isnan(res), res = fld.Value; end
