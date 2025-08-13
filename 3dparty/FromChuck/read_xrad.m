function [ im,head ] = read_xrad(fname)

if isempty(strfind(fname,'.header')), fname=[fname '.header']; end
fname
fid=fopen(fname,'rb');
header=fread(fid,inf,'char');
fclose(fid);

head=deblank(char(header)');
sz1=sscanf(head(strfind(head,'IDim'):end),'IDim=%d');
sz2=sscanf(head(strfind(head,'JDim'):end),'JDim=%d');
sz3=sscanf(head(strfind(head,'KDim'):end),'KDim=%d');
[pathstr,name,ext]=fileparts(fname);
imgfile=fullfile(pathstr,[name '.img'])
fid=fopen(imgfile,'rb');
im=fread(fid,inf,'single');
fclose(fid);
im=single(reshape(im,sz1,sz2,sz3));
sz=size(im)
end

