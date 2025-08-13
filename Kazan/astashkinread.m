% BRUKERREAD Read data from .dsc/.dta and .par/.spc data files
%
%   data = brukerreadJSS(filename)
%   [ax,data] = brukerreadJSS(filename);
%   [ax,data,dsc] = brukerreadJSS(filename);
%
%   Reads data and axis information from Bruker
%   files DSC/DTA and PAR/SPC which contains 'JSS' parameter. 
%   ax contains fields x,y,z 
%   depends on data dimension and may contain
%   other fields  as well
%   dsc is a cell array of description strings

% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelhaim an der Ruhr, 2003-2004
% Free for non-commercial use. Use the program at your own risk. 
% The authors retains all rights. 
% Contact: epel@mpi-muelheim.mpg.de

% bme,   4-dec-03, MPI
% allsi, 20-jan-04, MPI

function varargout = astashkinread(filename)

[fpath,fname] = fileparts(filename);
par_file = fullfile(fpath, [fname, '.par']);
dat_file = fullfile(fpath, [fname, '.dat']);

% load parameters
dsc = [];
fid = fopen(par_file, 'r');
while ~feof(fid)
  str = fgets(fid);
  if contains(str, ';'), continue; end
  pos = strfind(str, ':');
  if ~isempty(pos)
    option = strtrim(str(1:pos(end)-1));
    option(option == ' ') = '_';
    option(option == '[') = '';
    option(option == ']') = '';
    option(option == '.') = '';
    dsc.(option) = strtrim(str(pos(end)+1:end));
  end
end
fclose(fid);

dsc.KAZANformat = 'ASTASHKINPAR';

Xpts = str2double(dsc.X_Axis_Resolution_pts);
Ypts = str2double(dsc.Y_Axis_Resolution_pts);

fid = fopen(dat_file);
hdr = fread(fid, 8, 'uint8');
y = fread(fid, 'float');
fclose(fid);

h(1) = sum(hdr(1:4).*[0;256*256;256;1]);
h(2) = sum(hdr(5:8).*[0;256*256;256;1]);
disp(h)

ax.x = 1:Xpts;
ax.y = 1:Ypts;
if dsc.Field_Sweep_Coordinate =='Y'
 
elseif dsc.RF1_Sweep_Coordinate =='Y'
else
  ax.x = (str2double(dsc.Initial_delay) + str2double(dsc.Delay_step_X)*(0:Xpts-1))*1e-9;
  ax.xlabel = 'Time, s';
end

% requested
switch nargout
 case 1
   varargout = {y};
 case 2
   varargout = {ax, y};
 case 3
   varargout = {ax, y, dsc};
end
