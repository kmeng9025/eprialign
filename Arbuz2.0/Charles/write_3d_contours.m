function write_3d_contours(incell,fname)

ncons=size(incell,2);
fid=fopen(fname,'w');
for i=1:ncons
    x=cell2mat(incell(1,i));
    y=cell2mat(incell(2,i));
    npts=prod(size(x));
    fprintf(fid,'%d\n',npts);
    for j=1:5:npts
        arg=j:min([j+4 npts]);
        fprintf(fid,'%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f',x(arg),y(arg));
        fprintf(fid,'\n');
    end
end
fprintf(fid,'0\n');
fclose(fid);