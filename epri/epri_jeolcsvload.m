function [x,y,pars] = epri_jeolcsvload(fname)

the_fields = {'x-range min', 'xmin';...
          'x-range', 'xrange'; ...
          'micro frequency', 'rf'; ...
          'center field', 'cf'; ...
          'data length', 'length'...
          };
h = fopen(fname);
if h == -1
  error(fname);
end

while ~feof(h)
  s = fgets(h);
  
  for jj=1:size(the_fields, 1)
    res = regexp(s, ['\s*',the_fields{jj,1},'\s*=\s*(?<x>[\d\.]+)'], 'names');
    if ~isempty(res)
      pars.(the_fields{jj,2}) = str2double(res.x);
    end
  end

  res = regexp(s, '\s*data\[(?<n1>\d+)\.\.(?<n2>\d+)\]', 'names');
  if ~isempty(res)
    n1 = str2double(res.n1);
    n2 = str2double(res.n2);
    break;
  end
  
end

y = zeros(n2+1, 1);
pos = n1+1;
while ~feof(h)
  s = fgets(h);
  y(pos) = str2double(s);
  pos = pos + 1;
end

fclose(h);

x = linspace(pars.xmin, pars.xmin+pars.xrange, pars.length);

