function data = importXEPR1(filename)
% Function to import XEPR files into MatLab

% Strip off file extension if present using private extensionStrip function
filename = extensionStrip(filename);

% Try to open DSC file
fid1 = 0;
fid1 = openFile(filename,'DSC','');
if fid1 == -1
    data = -1;
    return
end

% Initialize values in structure 'NEX ' signifies nonexistent
data.byteSequence = 'NEX ';
data.realComplex = 'NEX ';
data.xTYPE = 'NEX ';
data.yTYPE = 'NEX ';
data.realFormat = 'NEX ';
data.imagFormat = 'NEX ';
data.xPTS = 0;
data.xMIN = 0;
data.xWIDTH = 0;
data.yPTS = 0;
data.yMIN = 0;
data.yWIDTH = 0;
data.TITLE = '';
data.iRealName = '';
data.iImagName = '';
data.xNAME = '';
data.yNAME = '';
data.iRealUnit = '';
data.iImagUnit = '';
data.xUNIT = '';
data.yUNIT = '';

% Read the DSC file line by line
while ~feof(fid1)
    % Read one line
    tline = fgetl(fid1);
    
    % Skip blank and short lines
    if(sum(size(tline)) <= 6)
        continue
    end

    % Trim white space from beginning of string
    tline = stringTrim(tline);
    
    % Skip comment lines
    if tline(1) == '#' | tline(1) == '*'
        continue
    end

    % Parse for keywords in the line
    switch upper(tline(1:4))
        
        % Check the byte sequence (Big/Little-endian)
       case 'BSEQ'
           tline = stringTrim(tline(5:end));
           if(sum(size(tline))-1 < 3)
               continue
           end
           switch upper(tline(1:3))
               case 'BIG'
                   data.byteSequence='ieee-be';
               case 'LIT'
                   data.byteSequence='ieee-le';
           end
           
        % Check for real/complex
       case 'IKKF'
           tline = stringTrim(tline(5:end));
            if(sum(size(tline))-1 < 4)
               continue
           end

           switch upper(tline(1:4))
               case 'REAL'
                   data.realComplex='REAL';
               case 'CPLX'
                   data.realComplex='CPLX';
           end
           
        % Check for X-Axis type   
       case 'XTYP'
          tline = stringTrim(tline(5:end));
           switch upper(tline(1:3))
               case 'IDX'
                   data.xTYPE='INDEXED';
               case 'IGD'
                   data.xTYPE='IGD';
               case 'NOD'
                   data.xTYPE='NoData';
           end
           
        % Check for Y-Axis type                 
       case 'YTYP'
          tline = stringTrim(tline(5:end));
           switch upper(tline(1:3))
               case 'IDX'
                   data.yTYPE='INDEXED';
               case 'IGD'
                   data.yTYPE='IGD';
               case 'NOD'
                   data.yTYPE='NoData';
           end
           
        % Check for Data Format (Real)                            
       case 'IRFM'
          tline = stringTrim(tline(6:end));
           switch upper(tline(1))
               case 'D'
                   data.realFormat='double';
               case 'F'
                   data.realFormat='float';
               case 'I'
                   data.realFormat='long';
           end
           
        % Check for Data Format (Imag)                                       
       case 'IIFM'
          tline = stringTrim(tline(6:end));
           switch upper(tline(1))
               case 'D'
                   data.imagFormat='double';
               case 'F'
                   data.imagFormat='float';
               case 'I'
                   data.imagFormat='long';
           end
           
        % Check for number of X points                                                  
       case 'XPTS'
           tline = stringTrim(tline(5:end));
           data.xPTS = str2num(tline);

        % Check for initial X value                                                  
       case 'XMIN'
           tline = stringTrim(tline(5:end));
           data.xMIN = str2double(tline);
           
        % Check for the X width                                                  
       case 'XWID'
           tline = stringTrim(tline(5:end));
           data.xWIDTH = str2double(tline);
           
        % Check for number of Y points                                                  
       case 'YPTS'
           disp('Hi!');
           tline = stringTrim(tline(5:end));
           data.yPTS = str2num(tline);

        % Check for initial Y value                                                  
       case 'YMIN'
           tline = stringTrim(tline(5:end));
           data.yMIN = str2double(tline);

        % Check for the Y width                                                  
       case 'YWID'
           tline = stringTrim(tline(5:end));
           data.yWIDTH = str2double(tline);
           
        % Check for the Title  
       case 'TITL'
           data.TITLE = deQuoter(tline);
           
        % Check for Real Intensity Name  
       case 'IRNA'
           data.iRealName = deQuoter(tline);
           
        % Check for Imag Intensity Name  
       case 'IINA'
           data.iImagName = deQuoter(tline);
           
        % Check for X Axis Name              
       case 'XNAM'
           data.xNAME = deQuoter(tline);
 
        % Check for Y Axis Name              
       case 'YNAM'
           data.yNAME = deQuoter(tline);
           
        % Check for Real Intensity Unit
       case 'IRUN'
           data.iRealUnit = deQuoter(tline);
           
        % Check for Imag Intensity Unit
       case 'IIUN'
           data.iImagUnit = deQuoter(tline);
           
        % Check for X Axis Unit              
       case 'XUNI'
           data.xUNIT = deQuoter(tline);
           
        % Check for Y Axis Unit              
       case 'YUNI'
           data.yUNIT = deQuoter(tline);

   end
end
fclose(fid1);

% Check for error conditions
errorHits = 0;

if(data.byteSequence(1:3) == 'NEX')
    disp('ERROR: No Byte Sequence specified in the DSC file')
    errorHits = errorHits + 1;
end

if(data.realComplex(1:3) == 'NEX')
    disp('ERROR: Real/Complex data not specified in the DSC file')
    errorHits = errorHits + 1;
end

if(data.xTYPE(1:3) == 'NEX')
    disp('ERROR: No X-axis type given')
    errorHits = errorHits + 1;
end

if(data.realFormat(1) == 'N')
    disp('ERROR: No data format specification for the intensity values')
    errorHits = errorHits + 1;
end

if((data.realComplex == 'CPLX') & (data.imagFormat(1) == 'N'))
    disp('ERROR: No data format specification for the imaginary data of a complex data set')
    errorHits = errorHits + 1;
end

if(data.xPTS == 0)
    disp('ERROR: No X points')
    errorHits = errorHits + 1;
end

if(data.xWIDTH == 0)
    disp('ERROR: No X Width')
    errorHits = errorHits + 1;
end

if((data.yTYPE(1) ~='N') & (data.yPTS <= 0))
    disp(data.yTYPE);
    disp(data.yPTS);
    disp('ERROR: no Y points for a 2D data set')
    errorHits = errorHits + 1;
end

if((data.yPTS ~= 0) & (data.yWIDTH == 0))
    disp('ERROR: No Y Width')
    errorHits = errorHits + 1;
end

% If errors occured, issue error message and return
if(errorHits > 0)
    disp(strcat(num2str(errorHits),' error(s) encountered with the DSC file'))
    data = errorHits;
    return
end

% Try to open DTA file
fid2 = 0;
fid2 = openFile(filename,'DTA',data.byteSequence);
if fid2 == -1
    data = -1;
    return
end

switch data.yPTS
    case 0  % Read in 1D data if 1D
        switch data.realComplex
            case 'REAL' % Method for Real data
                data.intensity = fread(fid2,data.realFormat);
                
            case 'CPLX' % Method for Complex data
                for k = 1:data.xPTS
                    data.intensity(k) = fread(fid2,1,data.realFormat) + i * fread(fid2,1,data.imagFormat);
                end
        end

        % Generate X-axis data for 1D data
        switch data.xTYPE
            case 'INDEXED'
                dx = data.xWIDTH/(data.xPTS-1);
                data.xValues(1:data.xPTS) = data.xMIN + (0:data.xPTS-1) * dx;
            case 'IGD'
                fid3 = 0;
                fid3 = openFile(filename,'XGF',data.byteSequence);
                if fid3 == -1
                    data = -1;
                    fclose(fid2)
                    return
                end
                data.xValues = fread(fid3,data.realFormat);
                fclose(fid3);
            otherwise
                disp('ERROR: Invalid xTYPE')
                data = -1;
                fclose(fid2);
                return
        end

                
        
    otherwise       % Read in 2D data if 2D
         switch data.realComplex     
            case 'REAL' % Method for Real data
                for n = 1:data.yPTS
                    for m = 1:data.xPTS
                        data.intensity(n,m) = fread(fid2,1,data.realFormat);
                    end
                end
                
            case 'CPLX' % Method for Complex data
                for n = 1:data.yPTS
                    for m = 1:data.xPTS
                        data.intensity(n,m) = fread(fid2,1,data.realFormat) + i * fread(fid2,1,data.imagFormat);
                    end
                end  
        end

        % Generate X-axis data for 2D data        
        switch data.xTYPE
            case 'INDEXED'
                dx = data.xWIDTH/(data.xPTS-1);
                for k = 1:data.yPTS
                    data.xValues(k,1:data.xPTS) = data.xMIN + (0:data.xPTS-1) * dx;
                end
            case 'IGD'
                fid3 = 0;
                fid3 = openFile(filename,'XGF',data.byteSequence);
                if fid3 == -1
                    data = -1;
                    fclose(fid2)
                    return
                end
                xValues = fread(fid3,data.realFormat);
                fclose(fid3);
                for k = 1:data.yPTS
                    data.xValues(k,1:data.xPTS) = xValues;
                end
                
            otherwise
                disp('ERROR: Invalid xTYPE')
                data = -1;
                fclose(fid2);
                return
        end
 
        % Generate Y-axis data for 2D data        
        switch data.yTYPE
            case 'INDEXED'
                dy = data.yWIDTH/(data.yPTS-1);
                for k = 1:data.xPTS
                    data.yValues(1:data.yPTS,k) = data.yMIN + (0:data.yPTS-1) * dy;
                end
            case 'IGD'
                fid3 = 0;
                fid3 = openFile(filename,'YGF',data.byteSequence);
                if fid3 == -1
                    data = -1;
                    fclose(fid2)
                    return
                end
                yValues = fread(fid3,data.realFormat);
                fclose(fid3);
                for k = 1:data.xPTS
                    data.yValues(1:data.yPTS,k) = yValues';
                end
                
            otherwise
                disp('ERROR: Invalid yTYPE')
                data = -1;
                fclose(fid2);
                return
        end
end

% Close DTA file
fclose(fid2);

function filename = extensionStrip(filename)
% Function to strip extensions from filenames

% Check for DSC extension: First eliminate case sensitivity
extension = findstr('.dsc',lower(filename));
% Check for no or multiple hits of .dsc
extensionSize = size(extension);
% If multiple hits: take the last value and truncate: E.g. If someone mistakenly names a
% file tdksjgh547y634.DSC.DSC
if(extensionSize(2) >= 1)
    filename = filename(1:extension(extensionSize(2))-1);
end

% Check for DTA extension: First eliminate case sensitivity
extension = findstr('.dta',lower(filename));
% Check for no or multiple hits of .dta
extensionSize = size(extension);
% If multiple hits: take the last value and truncate: E.g. If someone mistakenly names a
% file tdksjgh547y634.DTA.DTA
if(extensionSize(2) >= 1)
    filename = filename(1:extension(extensionSize(2))-1);
end

function fid = openFile(filename,fileType,byteSequence)
% Function to open up different filetypes
% Try to open DSC file

fid = 0;
fileType = upper(fileType);

switch fileType
    case 'DSC'
        fid=fopen(strcat(filename,'.DSC'),'r');
        if(fid == -1)
            % If unsuccessful: see if file extension is lower case: File transfer can
            % wreak havoc on case
            fid = fopen(strcat(filename,'.dsc'),'r');
            if(fid == -1)
                % If lower case doesn't work: file probably doesn't exist
                disp('ERROR: Can''t find the DSC file')
                return
            end
        end
    case {'DTA','XGF','YGF'}
        fid=fopen(strcat(filename,'.',fileType),'r',byteSequence);
        if(fid == -1)
            % If unsuccessful: see if file extension is lower case: File transfer can
            % wreak havoc on case
            fid = fopen(strcat(filename,'.',lower(fileType)),'r',byteSequence);
            if(fid == -1)
                % If lower case doesn't work: file probably doesn't exist
                disp(strcat('ERROR: Can''t find the ',fileType,' file'))
                return
            end
        end
    otherwise
        fid = -1;
        disp('ERROR: Invalid file type')
end

function stringOut = stringTrim(stringIn)
% Function to trim any whitespace character from the start of a string

firstLetter = 1;
sizeString = size(stringIn);

for k = 1:sizeString(2)
    if isletter(stringIn(k))
        firstLetter = k;
        break
    end
end
stringOut = stringIn(firstLetter:end);

function stringOut = deQuoter(stringIn)
% Function to extract text between quotes

% Find the quotes
quote = findstr('''',stringIn);

% If there aren't 2 quotes, something is wrong and set the string to the
% empty string
if size(quote) ~= 2
    stringOut = '';
    return
end

% Extract the text between the 2 quotes
stringOut = stringIn(quote(1)+1:quote(2)-1);

% If there is no text between the 2 quotes set the string to an empty
% string
if(sum(size(stringOut)) <= 1)
    stringOut = '';
end
