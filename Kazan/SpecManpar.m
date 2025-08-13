function res = SpecManpar(par, input_pars)

% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelhaim an der Ruhr, 2003
% Free for non-commercial use. Use the program at your own risk.
% The authors retains all rights.
% Contact: epel@mpi-muelheim.mpg.de

res.title = safeget(par, 'general_name', '?');
sweepax = {};
key = 'transient';
fullfield = ['sweep_', key];
idx = 0;
triggers = str2num(safeget(par, 'streams_triggers', '1'));
shots = 1;
scans = str2num(safeget(par, 'streams_scans', '1'));
while isfield(par, fullfield)
  [ax.t, str] = strtok(par.(fullfield), ',');
  ax.t = strtrim(ax.t); ax.t = ax.t(1);
  [ax.sSize, str] = strtok(str(2:end), ',');
  ax.dim = str2double(ax.sSize);
  ax.size = 1;
  ax.triggers = 1;
  if ax.t=='X' || ax.t=='Y' || ax.t=='Z' || ax.t=='T'
    ax.size = ax.dim;
  end
  [ax.reps, str] = strtok(str(2:end), ',');
  ax.reps = str2double(ax.reps);
  
  if ax.t == 'S'
    shots = shots * ax.dim * ax.reps;
  else
    shots = shots * ax.reps;
  end
  
  ax.var = {};
  while ~isempty(str)
    [ax.var{end+1}, str] = strtok(str(2:end), ',');
  end
  sweepax{end+1} = ax;
  fullfield = ['sweep_sweep', num2str(idx)];
  idx = idx +1;
end
sweepax{1}.triggers = triggers;
res.sweepax = sweepax;
res.shots = shots;
res.scans = scans(1);

axislabel = 'xyzijkl';
counter = 1;

sweepax2 = {};
for ii=1:length(sweepax)
  if sweepax{ii}.size > 1 || sweepax{ii}.triggers > 1
    sweepax2{end+1} = sweepax{ii};
  end
end

for k = 1:length(sweepax2)
  asize = sweepax2{k}.size;
  switch sweepax2{k}.t
    case {'I', 'A'}
      res.(axislabel(counter)) =  (1:sweepax2{k}.triggers)';
      res.([axislabel(counter), 'label']) = 'triggers';
    case 'T'
      if sweepax2{k}.triggers > 1
        res.(axislabel(counter)) =  (1:sweepax2{k}.triggers)';
        res.([axislabel(counter), 'label']) = 'triggers';
        if asize > 1
          counter = counter + 1;
        end
      end
      if asize > 1
        dwell_time_str = safeget(par, 'streams_dwelltime', '1 ns');
        dwell_time_str = strtrim(gettoken(dwell_time_str, ','));
        params_trans = ['0 ns step ', dwell_time_str,';'];
        values = get_array(params_trans, asize);
        res.(axislabel(counter)) =  values;
        res.([axislabel(counter), 'label']) = 'transient, s';
      end
    otherwise
      tempparam = sweepax2{k}.var{1};
      tempparam(tempparam == ' ') = '_';
      % check if this is a parameter
      if isfield(par, ['params_', tempparam])
        str = getSMfield(par, ['params_', tempparam]);
        values = get_array(str, asize);
        res.(axislabel(counter)) =  values;
        res.([axislabel(counter), 'label']) = tempparam;
      end
  end
  counter = counter + 1;
end
[res.StartTime, res.FinishTime, res.ExpTime] = GetSMTime(par);

if isfield(input_pars, 'Return')
  for ii=1:length(input_pars.Return)
    try
      strfield = input_pars.Return{ii};
      strvalue = par.(strfield{1});
      res.Return.(strfield{1}) = get_array(strvalue);
    catch
      disp('SpecManpar: Return was not processed.');
    end
  end
end


function [tk,rstr] = gettoken(istr, tok)

pos = strfind(istr, tok);

if isempty(pos)
  tk=istr;
  rstr='';
else
  tk=strtrim(istr(1:pos-1));
  rstr=strtrim(istr(pos+length(tok):end));
end

function str = getSMfield(dsc, fldname)

str = dsc.(fldname);
if contains(str, '*****')
  str = '';
  ii=0; fld_name_ext = [fldname, '_', num2str(ii)];
  while isfield(dsc, fld_name_ext)
    str1 = dsc.(fld_name_ext);
    if contains(str1, '*****'), break; end
    pos = strfind(str1, '*');
    str = [str, str1(pos(1)+1 : pos(end)-1)];
    ii=ii+1; fld_name_ext = [fldname, '_', num2str(ii)];
  end
end

function [StartTime, FinishTime, Time] = GetSMTime(dsc)
if ~isfield(dsc, 'general_starttime')
  StartTime = now; FinishTime = now; Time = 0;
  return;
end
% timestamp
str = strtrim(dsc.general_starttime);
t = textscan(str, '%s%s%s%s%s', 'Delimiter', ' ');
ML_date = [t{3}{1},'-',t{2}{1},'-',t{5}{1},' ',t{4}{1}];
StartTime = fix(datevec(ML_date));
str = strtrim(dsc.general_finishtime);
t = textscan(str, '%s%s%s%s%s', 'Delimiter', ' ');
ML_date = [t{3}{1},'-',t{2}{1},'-',t{5}{1},' ',t{4}{1}];
FinishTime = fix(datevec(ML_date));
Time = etime(FinishTime, StartTime)/60.;

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
else
  a = regexp(str, '\s*(?<data>[0-9.eE\-\+]+[\sa-z_A-Z]*\w*),*','names');
  val = [];
  for ii=1:length(a); val(end+1) = str2val(a(ii).data); end
end
val = val(:);
