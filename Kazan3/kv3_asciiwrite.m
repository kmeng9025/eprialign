% function varargout=kv3_asciiwrite(filename, src, varargin)
% is2d: 0/1 use dlmwrite instead of save
% useheader: 0/1 create header information
% usefooter: 0/1 create footer information
% origin: original filename
% parameters: kazan3 style of parameters see kv3_ParameterReader

function kv3_asciiwrite(filename, src, varargin)

opt.is2d = 0;
opt.useheader = 0;
opt.usefooter = 0;
opt.origin = '';
opt.parameters = {};

if nargin >= 2
  if ~mod(nargin-2,2)
    for kk=1:2:nargin-3
      opt.(lower(varargin{kk})) = varargin{kk+1};
    end
  else, error('Wrong amount of arguments')
  end
else, error('Usage: kv_asciiwrite(filename, src). Options: is2D, useheader, usefooter, origin, parameters.');
end

if opt.is2d
  p = real(src.y);
  dlmwrite(filename,p,' ')
else
  y  = real(src.y);
  y1 = imag(src.y);
  x  = src.ax.x;
  if sum(y1)
    p = [x,y,y1];
  else
    p = [x,y];
  end
  save(filename, 'p', '-ascii')
  
  if opt.useheader
  end
  
  if opt.usefooter
    
    fid = fopen(filename, 'a');
    fprintf(fid, ';--- created by Kazan Viewer 3.0 ---\n');
    fprintf(fid,'; original: %s\n', opt.origin);
    if isfield(src.ax, 'type'), fprintf(fid,'; ax.type: %s\n', src.ax.type); end
    if isfield(src.ax, 'xlabel'), fprintf(fid,'; ax.xlabel: %s\n', src.ax.xlabel); end
    if isfield(src.ax, 'ylabel'), fprintf(fid,'; ax.ylabel: %s\n', src.ax.ylabel); end
    if ~isempty(src.ax.filt)
      if isfield(src.ax, 'filt'), fprintf(fid,'; ax.filt: %s\n', src.ax.filt);  end
      if isfield(src.ax, 'tc'), fprintf(fid,'; ax.tc: %f\n', src.ax.tc); end
      if isfield(src.ax, 'pol'),fprintf(fid,'; ax.pol: %d\n', src.ax.pol); end
    end
    if ~isempty(src.ax.diff)
      if isfield(src.ax, 'diff'),fprintf(fid,'; ax.diff: %s\n', src.ax.diff); end
      if isfield(src.ax, 'ps_mod'), fprintf(fid,'; ax.ps_mod: %f\n', src.ax.ps_mod); end
    end
    if isfield(src.ax, 'dx'), fprintf(fid,'; ax.dx: %f\n', src.ax.dx); end
    if isfield(src.ax, 'dy'), fprintf(fid,'; ax.dy: %f\n', src.ax.dy); end
    if isfield(src.ax, 's'), fprintf(fid,'; ax.s: %f\n', src.ax.s); end
    if isfield(src, 'Script') % alsi 28.02.2005
      if length(src.Script) == 1
          fprintf(fid,'; Script: %s\n', src.Script{1});
      else
        fprintf(fid,'; Script:\n');
        for ci = 1:length(src.Script)
          fprintf(fid,'; %s\n', src.Script{ci});
        end
      end
    end
    
    fields = fieldnames(src.ax);
    for k=1:length(fields)
      fname = fields{k};
      idx = find(strcmp( cellfun( @(a) a.title, opt.parameters, 'UniformOutput', false ), {fname}), 1);
      if ~isempty(idx)
        switch opt.parameters{idx}.type
          case 's',  fprintf(fid,'; %s: %s\n', opt.parameters{idx}.title, src.ax.(fname));
          otherwise, fprintf(fid,'; %s: %f\n', opt.parameters{idx}.title, src.ax.(fname));
        end
      end
    end

    fclose(fid);
  end
end
