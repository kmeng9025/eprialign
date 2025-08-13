function fpath = epr_PathFromDate(Date, opt, folder)

if ischar(Date), Date = datenum(Date); end

vec = datevec(Date);
year = sprintf('%i', vec(1));
short_year = year(3:4);
smonth = sprintf('%02i', vec(2));
sdate = sprintf('%i', vec(3));
% data_root = '/Users/borisepel/Dropbox/matlab';
data_root = 'V:\';
% data_root = 'D:\';

switch opt
  case 'pulse250'
    fpath = fullfile(data_root, 'data', ...
      'Imagnet_PULSE', short_year, smonth, [short_year,smonth,sdate], folder);
  otherwise % 'imagnet'
    fpath = fullfile(data_root, 'data', ...
      ['Imagnet_data_', short_year], smonth, [short_year,smonth,sdate], folder);
end
