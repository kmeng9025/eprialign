function [ax, y, dsc] = kv_binary(filename, pars)
ax = [];
y = [];
dsc = [];

switch pars
  case 'i16'
    fid=fopen(char(filename),'r', 'ieee-le');
    A = fread(fid, 'int16=>double');
    fclose(fid);
  case 'i32'
    fid=fopen(char(filename),'r', 'ieee-le');
    A = fread(fid, 'int32=>double');
    fclose(fid);
end

if exist('A', 'var')
  sz = size(A);
  A = reshape(A, [2, sz(1)/2]);
  ax.x = A(1,:);
  y = A(2,:);
end




