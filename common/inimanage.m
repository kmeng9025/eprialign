function varargout = inimanage(varargin)
% outstruct = inimanage(filename)
% Reads and writes to 'ini'-files.
%    structure = inimanage(filename)
%    inimanage(filename, structure)
% The 'ini'-file style:
% [Options]
%     SingleValue1 = 122
%     SingleValue2 = hello
%     Array(1) = 123
%     Array(2) = 321
% Spaces in field and section names are not allowed.
% Single values are stored as strings, arrays as cell array of strings.
% AlSi 14.12.2004 for Kazan Viewer
switch nargin
    case 1
        varargout{1} = readini(varargin{1});
    case 2
        changeini(varargin{1}, varargin{2});
    otherwise
        disp('inimanage: number of input parameter is not matched')
end

%-------------------------------------------------------------------------
function outstruct = readini(varargin)
outstruct = [];
fname = varargin{1};
if ~ischar(fname), return; end
fid = fopen(fname, 'r');
if fid == -1, return; end
currstruct = 'noname';
outstruct.noname = [];
numcom = 1;
while ~feof(fid)
    a = fgets(fid);
    if ~ischar(a), break; end 
    tstr = strtrim(a); % to kill \n\r
    if isempty(tstr)
        % do nothing
    elseif strcmp(tstr(1), '[')
        currstruct = strtok(tstr(2:end), ']');
    elseif strcmp(tstr(1), '%')||strcmp(tstr(1), ';')
        eval(['outstruct.',currstruct,'.comment', num2str(numcom),' = strtrim(tstr(2:end));']);
        numcom = numcom+1;
    else
        [varname, varval] = strtok(tstr, '=');
        varname = strtrim(varname);
        bra = strfind(varname, '(');
        ket = strfind(varname, ')');
        if ~isempty(bra)&&~isempty(bra)
            numm = str2num(varname(bra+1:ket-1));
        else
            numm=0;
        end
        if ~isletter(varname(1)), flag = 'a'; 
        else,  flag = ''; end
        try
            if ~isfield(outstruct, currstruct), outstruct = setfield(outstruct, currstruct,{}); end            
            if ~numm
                eval(['outstruct.', currstruct, '.',flag,strtrim(varname), '=''', strtrim(varval(2:end)),''';']);
            else
                eval(['outstruct.', currstruct, '.', strtrim(varname(1:bra-1)),...
                        '{',varname(bra+1:ket-1) ,'}=''', strtrim(varval(2:end)),''';']);
            end
        catch
            fprintf('readini: I''ve read something wrong here: %s\n', tstr);
        end
    end
end
fclose(fid);


%-------------------------------------------------------------------------
function changeini(varargin)
fname = varargin{1};
instruct = varargin{2};

if isempty(instruct)
    instruct.noname = [];
end

chapter_names = fieldnames(instruct);
strings = {};
for cc = 1:length(chapter_names)
    if ~strcmp(chapter_names{cc}, 'noname')
        strings{end+1} = (['[',chapter_names{cc},']']);
    end
    chapter = getfield(instruct,(chapter_names{cc}));
    if isempty(chapter), continue; end
    var_names = fieldnames(chapter);
    for cj = 1:length(var_names)
        values = getfield(chapter,var_names{cj});
        if ~isempty(strfind(var_names{cj}, 'comment'))
            strings{end+1} = (['; ', values]);  
        else
            if ~iscell(values)
                strings{end+1} = ([var_names{cj},' = ', values]);    
            else
                for ci = 1:length(values)
                    strings{end+1} = ([var_names{cj},'(', num2str(ci),') = ', values{ci}]);    
                end
            end
        end
    end
    strings{end+1} = ' '; % Making spacing 
end

fid = fopen(fname, 'w');
for ca = 1:length(strings)
    fprintf(fid, '%s\r\n', strings{ca});
end
fclose(fid);

% ****************************************************************************************
% This function is absent in earlier versions of Matlab
function b = strtrim(b)

while ~isempty(b) && isspace(b(1))
    b = b(2:end);
end

while ~isempty(b) && isspace(b(end))
    b = b(1:end-1);
end
