function arr = kvgetarray(str)
[str1] = gettoken(str, ';'); % strip value part
arr = [];
[tk1, str1] = gettoken(str1, ',');
while ~isempty(tk1)
  [arr(end+1),unit] = kvgetvalue(tk1);
  [tk1, str1] = gettoken(str1, ',');
  if isempty(tk1) && ~isempty(str1)
    tk1 = str1; str1 = [];
  end
end

function [tk,rstr] = gettoken(istr, tok)

pos = strfind(istr, tok);

if isempty(pos)
    tk=istr;
    rstr='';
else
    tk=strtrim(istr(1:pos-1));
    rstr=strtrim(istr(pos+length(tok):end));
end