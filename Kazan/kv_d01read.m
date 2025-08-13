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

function varargout=kv_d01read(filename, varargin)

% convert varargin into structure
input_pars = [];
if nargin > 1
  for ii=1:2:nargin-1
    input_pars.(varargin{ii}) = varargin{ii+1};
  end
end

[path,name] = fileparts(filename);
fname = fullfile(path,[name,'.d01']);
dscname = fullfile(path,[name,'.exp']);

fid=fopen(char(fname),'r', 'ieee-le');
if fid<1, error(['File ''',fname,''' can not be open for read.']);end

ndim1=fread(fid, 1,'uint32');       % number of headers, re/im etc.
dformat=fread(fid,1,'uint32');     % format:0-double,1-float
if dformat==1
  sformat='float32';
else
  sformat='double';
end

dstrms = {};
ntotal = 1;
for k=1:ndim1
  ndim2       = fread(fid,1,'int32');
  dstrms{end+1}.dim = fread(fid,4,'int32');
  dstrms{end}.dim(ndim2+1:end) = 1;
  dstrms{end}.first = ntotal;
  dstrms{end}.total = fread(fid,1,'int32');
  ntotal = ntotal + dstrms{end}.total;
end
tmpdat=fread(fid,ntotal,sformat);
fclose(fid);

if ndim1 == 0
  error('No data present');
end

dsc = SpecMandsc(dscname);
ax = SpecManpar(dsc, input_pars);

if ndim1 > 1
  if all(dstrms{2}.dim == dstrms{1}.dim)
    spec=tmpdat(dstrms{1}.first:(dstrms{1}.first+dstrms{1}.total-1))+...
      1i*tmpdat((dstrms{2}.first:dstrms{2}.first+dstrms{2}.total-1));
    spec=reshape(spec,dstrms{1}.dim');
    for ii=3:ndim1
      ax.data{ii-2} = reshape(tmpdat(dstrms{ii}.first:(dstrms{ii}.first+dstrms{ii}.total-1)), ...
        dstrms{ii}.dim') / ax.scans(ii);
    end
  else
    spec = reshape(tmpdat(dstrms{1}.first:(dstrms{1}.first+dstrms{1}.total-1)), ...
      dstrms{1}.dim');
    for ii=2:ndim1
      ax.data{ii-2} = reshape(tmpdat(dstrms{ii}.first:(dstrms{ii}.first+dstrms{ii}.total-1)), ...
        dstrms{ii}.dim') / ax.scans(ii);
    end
  end
else
  spec=reshape(tmpdat,dstrms{1}.dim');
end

if ~isfield(ax, 'x') || size(ax.x, 1)~=size(spec, 1)
  ax.x = 1:size(spec, 1);
end
ax.type = 'data';

% timestamp
ax.RepTime = kvgetvalue(safeget(dsc, 'params_RepTime', '1 us'));

streams_scans  = kvgetvalue(safeget(dsc, 'streams_scans', '1,1,1,1'));

spec = spec / streams_scans;

% assign output depending on number of output arguments
% requested
dsc.KAZANformat = 'SPECMAND01';
switch nargout
 case 1
   varargout = {spec};
 case 2
   varargout = {ax, spec};
 case 3
   varargout = {ax, spec, dsc};
end
