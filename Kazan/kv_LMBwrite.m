function varargout=kv_LMBwrite(filename, src)

[path,name] = fileparts(filename);
fname = fullfile(path,[name,'.LMB']);
% dscname = fullfile(path,[name,'.DSC']);

sw = max(src.ax.x) - min(src.ax.x);
cf = (max(src.ax.x) + min(src.ax.x))/2;

[fid, ErrorMessage] = fopen(fname,'wb');
error(ErrorMessage);

fwrite(fid, 'ESR2');

Format = 'float32';
% fwrite(fid, 60, Format);  % [0] unused
fwrite(fid, single(sw), Format);  % [1] scan range in Gauss
fwrite(fid, single(cf), Format); % [2]field center in G.
fwrite(fid, length(src.y), Format); % number of data points.
fwrite(fid, 1, Format);  % [4] screen display x scale factor
fwrite(fid, 1, Format);  % [5] screen display y scale factor
fwrite(fid, 0, Format);  % [6] screen display x offset value, in data points
fwrite(fid, 0, Format);  % [7] screen display y offset value, in data points
fwrite(fid, 1, Format);  % [8] first data point to be displayed
fwrite(fid, length(src.y), Format);  % [9] last data point to be displayed
fwrite(fid, 20, Format); % [10] scan time in seconds
fwrite(fid, 0, Format);  % [11] FT counter, 0= cw spectrum
fwrite(fid, 0, Format);  % [12] FT combine value, for combine Real and Imag. components
fwrite(fid, 0, Format);  % [13] split (double) scan Y offset, from 1 to 9, screen units
fwrite(fid, 0, Format);  % [14] split scan gain, relative to main scan
fwrite(fid, 0, Format);  % [15] split scan delay (X offset) in data points
fwrite(fid, 0, Format);  % [16] unused
fwrite(fid, 0, Format);  % [17] unused
fwrite(fid, 0, Format);  % [18] unused
fwrite(fid, 0, Format);  % [19] unused
fwrite(fid, 0, Format);  % [20] unused

fwrite(fid, single(src.y(:)), Format); % output data

fwrite(fid, sprintf('%-60s', 'hello'), 'char'); % output comment
fpars{1} = sprintf('%-12s', 'hello');
fpars{2} = sprintf('%-12s', '0.5G');
fpars{3} = sprintf('%-12s', '100.0 khz');
fpars{4} = sprintf('%-12s', '1.0');
fpars{5} = sprintf('%-12s', '1e+5');
fpars{6} = sprintf('%-12s', '60.000');
fpars{7} = sprintf('%-12s', '3325.040');
fpars{8} = sprintf('%-12s', '100.0');
fpars{9} = sprintf('%-12s', '9.33GHz');
fpars{10} = sprintf('%-12s', '3-12-94');
fpars{11} = sprintf('%-12s', '12:05');
fpars{12} = sprintf('%-12s', '240.0');
fpars{13} = sprintf('%-12s', '1');
fpars{14} = sprintf('%-12s', 'T');
fpars{15} = sprintf('%-12s', 'M');
fpars{16} = sprintf('%-12s', '');
fpars{17} = sprintf('%-12s', '');
fpars{18} = sprintf('%-12s', '');
fpars{19} = sprintf('%-12s', '');
fpars{20} = sprintf('%-12s', '');
for ii=1:20
  fwrite(fid, fpars{ii}); % ouput strings
end

fwrite(fid, char(60)); % output extra comments
fwrite(fid, char(60));

fclose(fid);

% reader

% f = 'c:\Temp\WinEPR\eprdata\dmpo.lmb';
% f = 'C:\Temp\WinEPR\Y2O3_0%_1.LMB';
% 
% fid = fopen(f, 'rb');
% a    = fread(fid, 4, 'uint8=>char')'; disp(a);
% pars = fread(fid, 20, 'float32')'
% 
% data = fread(fid, pars(3), 'float32');
% a    = fread(fid, 60, 'uint8=>char')'; disp(a);
% for ii=1:20
%   
%   a    = fread(fid, 12, 'uint8=>char')'; disp([num2str(ii), ':', a]);
% end
% a    = fread(fid, 60, 'uint8=>char')'; disp(a);
% a    = fread(fid, 60, 'uint8=>char')'; disp(a);
% fclose(fid);
% 
% figure; plot(data)