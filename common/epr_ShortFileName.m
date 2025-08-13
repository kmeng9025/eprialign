function strout = epr_ShortFileName(strin, nchars)
% function strout = epr_ShortFileName(strin, nchars)
% Make file name shorter by removing the elements of the path

slash = sort([findstr(strin, '\'),findstr(strin, '/')]);

if isempty(slash) || length(strin) <= nchars
    strout = strin;
else
    strout = strin(slash(end)+1:end);
    add_chars = nchars - length(strout);
    if add_chars > 6
       strout = [strin(1:3),'...',strin(slash(end)+ [-add_chars+3:0]),strout]; 
    end
end