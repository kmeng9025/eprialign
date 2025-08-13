function [h,SP,Nh,Hpp,Fm] = readBES3Tm(fname)
%% Help
% It is Matlab version of a Octave file Bruker sent to me. 
% h  - magnetic filed vector
% SP -2D dataset
% Nh - number of harmonic signals  both in and out of phase
%% Extention check
Nc=numel(fname); % number of characters in the fname
ext=fname(end-3:end);    % Extention 
Gname=fname(1:end-4);   % Global fine name (without ext)
switch ( lower(ext) )
        case '.dsc'
            parname = fname;           % par file
            spcname = [Gname '.DTA' ]; % binary file
        case '.dta'
            parname = [Gname '.DSC' ];
            spcname = fname;
end
%% Read par file 
fid = fopen( parname, 'r' );
if ( fid < 0 ); error( ['Cannot file open ' parname ]);  end;
line='';
while ischar(line)
line=fgetl(fid);
[key,val] = strtok(line);
%disp(key)
 	switch ( key )
 	case 'XMIN'
 	    XMIN = str2num(val);
 	case 'XWID'
 	    XWID =str2num(val);
 	case 'XPTS'
 	    XPTS = str2num(val);
 	case 'B0MA'
 	    Hpp = 10000*str2num(val);
     case 'ModFreq'
 	    Fm = val;
        
 	end
 end
fclose( fid );
h=linspace( XMIN, XMIN+XWID, XPTS ); % Magnetic Fields

%% Read binary file
fid = fopen( spcname, 'r', 'ieee-be' );
if ( fid < 0 ); error( ['Cannot file open ' parname ]);  end;
sp = fread( fid, 'float64' );
M=length(sp);
Nh=M/XPTS; % Number of harmonic spectra 
SP=reshape(sp,Nh,XPTS);
fclose( fid );
