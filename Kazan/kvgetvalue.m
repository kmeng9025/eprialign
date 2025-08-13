% KVGETVALUE read string value in ci-standard units
% KAZAN viewer routine
% 
% [val, str_unit, str_koefficient] = kvgetvalue(str)

% boep 14-mar-04, MPI for Bioinorganic Chemistry 


function [val, unit, pref, pref_val] = kvgetvalue(str)

prefix = ['p','n', 'u', 'm', 'k', 'M', 'G', 'T'];
koeff  = [1E-12, 1E-9, 1E-6, 1E-3, 1E3, 1E6, 1E9, 1E12];
pref = ''; pref_val = 1;

res = regexp(str, '(?<number>[\d\.\-\E\e]+)\s*(?<unit>\w+)*', 'names');

if ~isempty(res)
  res = res(1);
  val = str2double(res.number);
  if isfield(res, 'number'), unit = char(res.unit); else unit = ''; end
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
