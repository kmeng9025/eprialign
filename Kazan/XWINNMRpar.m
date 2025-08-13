% reading Bruker nmr header of .acq type

function par = XWINNMRpar(filename)
h = fopen(filename);
olda = '';
par = [];
while feof(h)<1
  s = fgetl(h);
  if strfind(s,'##')
    if strfind(s,'$$'), continue; end; 
    s(s=='#') = ' ';
    [fld, s]=strtok(strtrim(s),'=');
    fld = fld(fld~='$');
    if ~isempty(fld)
      par = setfield(par, fld, strtrim(s(2:end)));
    end
  else
    if ~isempty(fld)
      par = setfield(par, fld, [getfield(par,fld),'; ',strtrim(s)]);
    end
  end
end
fclose(h);
