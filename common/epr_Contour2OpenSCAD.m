function epr_Contour2OpenSCAD(mod_name, fname, vectors, scale)



endl = sprintf('\n');
str = ['//',mod_name,'(10);', endl, endl];
str = [str,'module ',mod_name,'(h)', endl];
str = [str,'{',endl];
str = [str, sprintf(' scale([%g, %g, %g])', scale(1), scale(2), scale(3)),endl];
str = [str,' union() {', endl];

for k = 1:length(vectors)
  
  str = [str,'linear_extrude(height=h) polygon([', endl];
  vector = vectors{k};
  
  separator = '';
  for i=1:size(vector,1)
    str = [str,separator,sprintf('[%g,%g]',vector(i,1),vector(i,2))];
    separator = ',';
  end
  str = [str,']);', endl];
end

str = [str,'}',endl];
str = [str,'}',endl];


fid = fopen(fname, 'w+');
fwrite(fid, str);
fclose(fid);
