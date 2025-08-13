function res = SpecManTDMSpar(par)

% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelhaim an der Ruhr, 2003
% Free for non-commercial use. Use the program at your own risk. 
% The authors retains all rights. 
% Contact: epel@mpi-muelheim.mpg.de


prefix = ['n', 'u', 'm', 'k', 'M', 'G'];
koeff  = [1E-9, 1E-6, 1E-3, 1E3, 1E6, 1E9];
res.title = safeget(par.root, 'name', '?');
sweepax = {};
key = 'transient';
fullfield = key;
idx = 0;

reference_stream = par.streams.Re;
triggers = safeget(reference_stream, 'triggers', 1);
shots = 1;
while isfield(par.axis, fullfield)
    [ax.t, str] = strtok(par.axis.(fullfield), ',');
    ax.t = strtrim(ax.t); ax.t = ax.t(1);
    [ax.size, str] = strtok(str(2:end), ',');
    ax.dim = str2double(ax.size);
    if ax.t=='S' || ax.t=='I' || ax.t=='A' || ax.t=='R', ax.size = 1; else  ax.size = ax.dim; end
    [ax.reps, str] = strtok(str(2:end), ',');
    ax.reps = str2num(ax.reps);

    if ax.t == 'S', shots = shots * ax.dim * ax.reps;
    else shots = shots * ax.reps;
    end
    
    ax.var = {};
    while ~isempty(str)
        [ax.var{end+1}, str] = strtok(str(2:end), ',');
    end
    sweepax{end+1,1} = ax;
    fullfield = ['sweep', num2str(idx)];
    idx = idx +1;
end
sweepax{1}.size=sweepax{1}.size*triggers;
res.sweepax = sweepax;
res.shots = shots;

axislabel = 'xyzabc';
counter = 1;
for k = 1:size(sweepax, 1)
    arr = [];
    asize = double(sweepax{k}.size);
    if asize > 1
        switch sweepax{k}.t
            case {'I', 'A'}
                tempparam = 'trans';
                par.params.trans = '1sl step 1sl;';
            case 'T'
                tempparam = 'trans';
                dwell_time = safeget(reference_stream, 'dwelltime', 1E-9);
                par.params.trans = sprintf('0 ns step %f ns', dwell_time*1E9);
            otherwise
                tempparam = sweepax{k}.var{1};
                tempparam(findstr(tempparam, ' ')) = '_';
        end
        % check if this is a parameter
        if isfield(par.params, tempparam)
            str = par.params.(tempparam);
            if contains(str,'step')
                [tk1, str1] = gettoken(str, 'step');
                % string of the type 10ns step 6 ns
                tk2 = strtrim(gettoken(str1, ';'));
                [minval, unit] = kvgetvalue(tk1);
                step = kvgetvalue(tk2);
                arr = double((0:asize-1)*step+minval);
            elseif contains(str,'logto')
                [tk1, str1] = gettoken(str, 'logto');
                % string of the type 10ns logto 60 ns
                tk2 = strtrim(gettoken(str1, ';'));
                [minval, unit] = kvgetvalue(tk1);
                maxval = kvgetvalue(tk2);
                arr = logspace(log10(minval), log10(maxval),asize);
            elseif contains(str,'to')
                [tk1, str1] = gettoken(str, 'to');
                % string of the type 10ns to 60 ns
                tk2 = strtrim(gettoken(str1, ';'));
                [minval, unit] = kvgetvalue(tk1);
                maxval = kvgetvalue(tk2);
                arr = linspace(minval, maxval, asize);
            else
                % string of the type 10ns, 20ns, 30ns;
                pos = strfind(str, ':');
                if ~isempty(pos)
                  inp = str(pos+1:end);
                  inp(inp == ',') = ' ';
                  arr = textscan(inp, '%f');
                  arr = arr{1};
                else
                [str1] = gettoken(str, ';');
                [tk1, str1] = gettoken(str1, ',');
                while ~isempty(tk1)
                    [arr(end+1),unit] = kvgetvalue(tk1);
                    [tk1, str1] = gettoken(str1, ',');
                    if isempty(tk1) && ~isempty(str1)
                        tk1 = str1; str1 = [];
                    end
                end
                end
            end
        else
            str = getfield(par.aquisition, tempparam);
            arr = [0:asize-1]';
            unit = 's';
        end
        
        % Unit normalization
        switch unit
          case 'G'
          case 'K'
          case 's'
          case ''
          otherwise
            umax = max(abs(arr));
            for kk = length(koeff):-1:1
              if umax > koeff(kk)
                uk = koeff(kk);
                unit = [prefix(kk), unit];
                arr = arr./uk;
                break;
              end
            end
        end
        
        if length(arr) > sweepax{k}.size, arr = arr(1:sweepax{k}.size); end
        res = setfield(res, axislabel(counter), arr');
        res = setfield(res, [axislabel(counter), 'label'], ...
            [sweepax{k}.var{1}, ', ',unit]);
        counter = counter + 1;
    end
end
res.StartTime = par.root.starttime;
res.FinishTime = par.root.finishtime;
res.ExpTime = par.root.totaltime;
res.RepTime = get_val(safeget(par.params, 'RepTime', '15 us'));

% --------------------------------------------------------------------
function [tk,rstr] = gettoken(istr, tok)

pos = strfind(istr, tok);

if isempty(pos)
    tk=istr;
    rstr='';
else
    tk=strtrim(istr(1:pos-1));
    rstr=strtrim(istr(pos+length(tok):end));
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

