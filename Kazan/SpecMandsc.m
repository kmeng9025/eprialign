function par = SpecMandsc(filename)
h = fopen(filename);
if h<0
  disp('Description file was not found.');
  par = [];
  return;
end
par = [];
forbidden = '~!@#$%^&*()./\';
section = '';
text = 1; prg = 1;
while feof(h)<1
  s = strtrim(fgetl(h));
  if isempty(s), continue; end;
  try
  sect = find(s=='[' | s==']');
  %   this is a section header   
  if size(sect, 2)==2 && sect(1)==1
    section = s(sect(1)+1:sect(2)-1);
%     par = setfield(par, section, 'section');
  else
    switch section
      case 'text'
        par = setfield(par, ['text', num2str(text)], s);
        text = text + 1;
      case 'program'
        par = setfield(par, ['prg', num2str(prg)], s);
        prg = prg + 1;
      otherwise
        [a,s]=strtok(s, '=');
        a = strtrim(a);
        a(a=='/' | a=='\' | a==' ' | a=='.')='_';
        if section(1) >= '0' && section(1) <= '9'
          section = ['f', section];
        end
        par = setfield(par, [section,'_',a], strtrim(s(2:end)));
    end  
  end
  catch e
    disp(sprintf('Error: %s in %s', e.message, s))
  end
end
fclose(h);
