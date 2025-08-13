% kv_d01read Read data from .d01/.exp SpecMan data files
%
%   data = kv_d01read(filename,...)
%   [ax,data] = kv_d01read(filename,...);
%   [ax,data,dsc] = kv_d01read(filename,...);
%
%   Reads data and axis information from Bruker
%   files  DSC/DTA  and  PAR/SPC.   Aditional 
%   arguments  are  always  coming  in pairs 
%   field, value. Fields are:
%     'Dataset', [real dataset, imag dataset] 
%   ax contains fields x,y,z  depends on data 
%   dimension and may contain other fields as 
%   well. dsc is a cell array  of description 
%   strings.

% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelheim an der Ruhr, 2003
% Free for non-commercial use. Use the program at your own risk. 
% The authors retains all rights. 
% Contact: epel@mpi-muelheim.mpg.de

% Igor Gromov, ETH-Hoenggerberg, 17.01.03
% bme,   4-dec-03, MPI  

function varargout=kv_d02read(filename, varargin)

[path,name] = fileparts(filename);
fname = fullfile(path,[name,'.d02']);
dscname = fullfile(path,[name,'.exp']);

dsc = SpecMandsc(dscname);
ax = SpecManpar(dsc);

sz = 1;
hidden_dim = 1; % dimensions of sum axis

for ii=1:length(ax.sweepax), sz = sz * ax.sweepax{ii}.size; end
channels = length(ax.sweepax{1}.var); 

for ii=1:length(ax.sweepax)
  if ax.sweepax{ii}.t == 'S'
    hidden_dim = hidden_dim * ax.sweepax{ii}.dim;
  end
end

xax = ax.x; xsh = max(xax)-min(xax);
for ii=1:hidden_dim - 1
  ax.x = [ax.x; xax + ii*xsh*1.1];
end

fid=fopen(char(fname),'r', 'ieee-le');
if fid<1, error(['File ''',fname,''' can not be open for read.']);end

fileinfo = dir(filename);
trans = 1;
points = fix(fileinfo.bytes/4/(1+trans));
full_scans = points/sz/channels/hidden_dim;
full_size  = sz*hidden_dim;
idx= zeros(points, 1); 
data=zeros(points, 1);
for pos=1:points
  idx(pos)=fread(fid, 1,'uint32');   
  data(pos)=fread(fid,1,'float');
end
fclose(fid);
idx = idx + 1; % convert into MATLAB style

if hidden_dim > 1
% separate indexes for different phase cycling of the type I S X
% will fail for I X S
  idx = idx + repmat((0:hidden_dim-1)*sz, 1, full_scans*channels*sz)';
end

idx = reshape(idx, [channels*full_size, full_scans]);
idx = idx + repmat((0:full_scans-1)*channels*full_size, channels*full_size, 1);
data_tmp = zeros(channels*full_size*full_scans, 1);
data_tmp(idx(:)) = data;
data_tmp = reshape(data_tmp, [sz*hidden_dim, channels, full_scans]);
rbuffer = squeeze(data_tmp(:,1,:)); 
switch channels
  case 2, ibuffer=squeeze(data_tmp(:,2,:)); spec = rbuffer(:)+ 1i * ibuffer(:);
  case 1, spec = rbuffer(:);
end

spec = reshape(spec, full_size, full_scans);
ax.y = (1:full_scans)';
ax.ylabel = 'scans';

% assign output depending on number of output arguments
% requested
switch nargout
 case 1,
   varargout = {spec};
 case 2,
   varargout = {ax, spec};
 case 3,
   varargout = {ax, spec, dsc};
end
