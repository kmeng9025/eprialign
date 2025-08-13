function res = kv_show(value, level)
res = '';
if ~exist('level', 'var'), level = 0; end
if level == 0, disp('------'); end
offset = '';
for ii=1:level, offset = [offset, ' ']; end

if isstruct(value)
    fnames = fieldnames(value);
    for ii=1:length(fnames)
        newvalue = value.(fnames{ii});
        res2 =  [fnames{ii}, ': '];
        if length(newvalue) > 1
            for jj=1:length(newvalue)
                res1 = sprintf('%i| %s || ', jj, kv_show(newvalue(jj), level+1));
                res2 = [res2, res1];
            end
            res1 = res2;
        else
            res1 = sprintf('%s%s ', res2, kv_show(newvalue, level+1));
        end
        if level == 0
            disp(res1);
        else
            res = [res, res1];
        end
    end
elseif iscell(value)
    res = 'cell';
elseif ischar(value)
    res = ['"', value, '"'];
    if level == 0, disp(res); end
else
    res = sprintf('%g ', value(:));
    if level == 0, disp(res); end
end
if level == 0, disp('------'); end
