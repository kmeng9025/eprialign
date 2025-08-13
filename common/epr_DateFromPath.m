% [fdate, fnewpath] = epr_DateFromPath(fpath, path_style)
% Convert LFEPR style file path to date in YYMMDD style
% Back conversion to directory path of other style is optional
% path_style can be: 'Boris_G', 'root_new', 'root_old'
function [fdate, fnewpath, ret] = epr_DateFromPath(fpath, path_style, root_path)

ret = [];
path_suffix = '';

% long YYYYmmDD string
res = regexp(fpath, '(?<date>\d\d\d\d\d\d\d\d).*', 'names');
if ~isempty(res) && ~isempty(res.date)
  fdate = res.date(3:end);
  the_yearYYYY = res.date(1:4);
  the_yearYY  = res.date(3:4);
  the_monthMM = res.date(5:6);
  the_dayDD = res.date(7:8);
else
  % YYmmDD string
  res = regexp(fpath, '(?<date>\d\d\d\d\d\d)(?<suffix>.*)', 'names');
  if ~isempty(res) && ~isempty(res.date)
    fdate = res.date;
    the_yearYY = fdate(1:2);
    the_yearYYYY = ['20', the_yearYY];
    the_monthMM = fdate(3:4);
    the_dayDD = fdate(5:6);
    path_suffix = res.suffix;
  else
    fdate = date;
    the_yearYY = fdate(10:11);
    the_yearYYYY = fdate(8:11);
    the_monthMM = fdate(4:6);
    the_dayDD = fdate(1:2);
    path_suffix = '';
  end
end


if exist('path_style', 'var')
  switch path_style
    case 'Boris_G'
      fnewpath = ['D:\ProcessedData\', the_yearYYYY,'\', fdate, '\'];
    case 'root_new'
      fnewpath = fullfile(root_path, [the_yearYYYY,'\', the_yearYYYY, the_monthMM, the_dayDD, path_suffix]);
    case 'root_old'
      fnewpath = fullfile(root_path, [the_yearYY,'\', the_monthMM, '\', the_yearYY, the_monthMM, the_dayDD, path_suffix]);
    otherwise
      fnewpath = fdate;
  end
else
  fnewpath = fdate;
end

try
  ret.datenum = datenum(str2double(the_yearYYYY),str2double(the_monthMM),str2double(the_dayDD));
  ret.datestr = datestr(ret.datenum);
  ret.folder = path_suffix;
  ret.folder(ret.folder == '\') = [];
catch
  ret.datenum = '';
  ret.datestr = '';
  ret.folder = '';
end
