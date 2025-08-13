% function varargout=kv_brukerwrite(filename, src, varargin)
% arguments 
%    format    double/float 
function varargout=kv_brukerwrite(filename, src, ftype,varargin)

opt = [];
if nargin > 2
  if ~mod(nargin-3,2)
    for kk=1:2:nargin-4
      opt=setfield(ax, lower(varargin{kk}), varargin{kk+1});
    end
  else
    error('Wrong amount of arguments')
  end
end
if(~isfield(src, 'dsc')), src.dsc = []; end

switch upper(ftype)
  case 'DSC'
    [path,name] = fileparts(filename);
    fname = fullfile(path,[name,'.DTA']);
    dscname = fullfile(path,[name,'.DSC']);
    
    BSEQ = safeget(src.dsc, 'BSEQ', 'BIG');
    
    if strcmp(BSEQ,'BIG'), ByteOrder = 'ieee-be';
    else, ByteOrder = 'ieee-le';
    end
    switch safeget(src.dsc,'IRFMT','D')
      case 'I',Format = 'int32';
      case 'I32',Format = 'float32';
      otherwise, Format = 'float64';
    end
    
    Dims = size(src.y);
    Complex = safeget(src.ax, 'complex', 1);
    
    % open data file
    [fid, ErrorMessage] = fopen(fname,'wb',ByteOrder);
    error(ErrorMessage);
    % calculate expected number of elements and read in
    N = prod(Dims);
    
    % reshape to matrix and permute dimensions if wanted
    out = reshape(src.y, [1,N]);
    
    % convert to complex
    if Complex
      out = reshape([real(out);imag(out)], [2*N,1]);
    end
    
    effN = fwrite(fid,out,Format);
    if effN<N
      error('Unable to write all expected data.');
    end
    
    % close file
    St = fclose(fid);
    if St<0, error('Unable to close data file.'); end
    
    
    % units and scaling
    xlb = safeget(src.ax, 'xlabel', '');
    pos = strfind(xlb,',');
    namex = xlb(1:pos-1);
    uni = xlb(pos+1:end);
    unitx = uni(uni~=' ');
    switch unitx
      case 's', namex = 'Time'; unitx = 'ns'; unitxc = 1E9;
      otherwise
        unitxc = 1.0;
    end
    
    Dims(end+1:3) = 1;
    if Dims(1) > 1, XTYP = 'IDX'; else, XTYP = 'NODATA'; end
    if Dims(2) > 1, YTYP = 'IDX'; else, YTYP = 'NODATA'; end
    if Dims(3) > 1, ZTYP = 'IDX'; else, ZTYP = 'NODATA'; end
    if Complex > 0, IKKF = 'CPLX'; else, IKKF = 'REAL'; end
    
    fid = fopen(char(dscname), 'w');
    fprintf(fid, '#DESC	1.2 * DESCRIPTOR INFORMATION ***********************\n');
    fprintf(fid, '*\n');
    fprintf(fid, '*	Dataset Type and Format:\n');
    fprintf(fid, '* \n');
    fprintf(fid, 'DSRC %s\n', safeget(src.dsc, 'DSRC', 'EXP'));
    fprintf(fid, 'BSEQ %s\n', safeget(src.dsc, 'BSEQ', BSEQ));
    fprintf(fid, 'IKKF %s\n', IKKF);
    fprintf(fid, 'XTYP %s\n', XTYP);
    fprintf(fid, 'YTYP %s\n', YTYP);
    fprintf(fid, 'ZTYP %s\n', ZTYP);
    
    fprintf(fid, '*\n');
    fprintf(fid, '*	Item Formats:\n');
    fprintf(fid, '* \n');
    fprintf(fid, 'IRFMT	D\n');
    fprintf(fid, 'IIFMT	D\n');
    fprintf(fid, '*\n');
    fprintf(fid, '*	Data Ranges and Resolutions:\n');
    fprintf(fid, '* \n');
    x = src.ax.x*unitxc;
    fprintf(fid, 'XPTS %d\n', Dims(1));
    fprintf(fid, 'XMIN %f\n', min(x));
    fprintf(fid, 'XWID %f\n', max(x) - min(x));
    if Dims(2) > 1
      fprintf(fid, 'YPTS %d\n', Dims(2));
      fprintf(fid, 'YMIN %f\n', min(src.ax.y));
      fprintf(fid, 'YWID %f\n', max(src.ax.y) - min(src.ax.y));
    end
    
    fprintf(fid, '*\n');
    fprintf(fid, '*	Documentational Text:\n');
    fprintf(fid, '* \n');
    % strip ' symbols
    title = safeget(src.ax, 'title', '');
    title  = title(title~='''');
    fprintf(fid, 'TITL ''%s''\n', title);
    fprintf(fid, 'IRNAM	''%s''\n','Intensity');
    fprintf(fid, 'IINAM	''%s''\n','Intensity' );
    xlb = safeget(src.ax, 'xlabel', '');
    pos = strfind(xlb,',');
    fprintf(fid, 'XNAM	''%s''\n', namex);
    fprintf(fid, 'IRUNI	''%s''\n','');
    fprintf(fid, 'IIUNI	''%s''\n','');
    fprintf(fid, 'XUNI	''%s''\n',unitx);
    ylb = safeget(src.ax, 'ylabel', '');
    if Dims(2) > 1
      fprintf(fid, 'YNAM	%s\n', ylb);
      fprintf(fid, 'YUNI	%s\n','');
    end
    fprintf(fid, '*\n');
    fprintf(fid, '************************************************************\n');
    fprintf(fid, '*\n');
    fprintf(fid, '#SPL	1.2 * STANDARD PARAMETER LAYER\n');
    fprintf(fid, '*\n');
    fprintf(fid, 'OPER    KazanViewer\n');
    fprintf(fid, 'DATE    %s\n', datestr(date,'dd/mm/yy'));
    fprintf(fid, 'TIME    %s\n', datestr(now,'HH:MM:SS'));
    fprintf(fid, 'CMNT    %s\n', safeget(src.ax, 'comment', ''));
    fprintf(fid, 'SAMP    %s\n', safeget(src.ax, 'sample', ''));
    fprintf(fid, 'SFOR    %s\n', safeget(src.ax, 'formula', ''));
    % fprintf(fid, 'STAG    A\n');
    % fprintf(fid, 'EXPT    CW\n');
    % fprintf(fid, 'OXS1    IADC\n');
    % fprintf(fid, 'AXS1    B0VL\n');
    % fprintf(fid, 'AXS2    NONE\n');
    % fprintf(fid, 'AXS3    \n');
    % fprintf(fid, 'A1CT    0.71\n');
    % fprintf(fid, 'A1SW    1.4\n');
    fprintf(fid, 'MWFQ    %8.1f\n', safeget(src.ax, 'freq1', 0));
    % fprintf(fid, 'MWPW    0.06331\n');
    % fprintf(fid, 'AVGS    2\n');
    % fprintf(fid, 'RESO    DM9509\n');
    % fprintf(fid, 'SPTP    0.08192\n');
    % fprintf(fid, 'RCAG    60\n');
    % fprintf(fid, 'RCHM    1\n');
    % fprintf(fid, 'B0MA    0.001\n');
    % fprintf(fid, 'B0MF    100000\n');
    % fprintf(fid, 'RCPH    0.0\n');
    % fprintf(fid, 'RCOF    -6.0\n');
    % fprintf(fid, 'A1RS    8192\n');
    % fprintf(fid, 'RCTC    0.08192\n');
    % fprintf(fid, 'STMP    9.99 \n');
    
    fprintf(fid, '*  Generated by Kazan Viewer 2.5\n');
    
    fclose(fid);
%****************************************************************
%****************************************************************
%****************************************************************
%****************************************************************
%****************************************************************
  case 'PAR'
    [path,name] = fileparts(filename);
    fname = fullfile(path,[name,'.par']);
    dscname = fullfile(path,[name,'.spc']);
    
    N = numel(src.y);
    
    % open data file
%     ByteOrder = 'ieee-le';
    ByteOrder = 'ieee-be';
    Format = 'int32';
%     Format = 'float32';
    [fid, ErrorMessage] = fopen(fname,'wb',ByteOrder);
    error(ErrorMessage);
    effN = fwrite(fid, src.y(:),Format);
    if effN<N
      error('Unable to write all expected data.');
    end
    fclose(fid);

    fid = fopen(char(dscname), 'w');
    fprintf(fid, 'JSS    0\n');
    fprintf(fid, 'JON    \n');
    fprintf(fid, 'JRE    SHQ\n');
    fprintf(fid, 'JDA    \n');
    fprintf(fid, 'JTM    \n');
    fprintf(fid, 'JCO    \n');
    fprintf(fid, 'JUN    Gauss\n');
    fprintf(fid, 'JNS    1\n');
    fprintf(fid, 'JSD    0\n');
    fprintf(fid, 'JEX    EPR\n');
    fprintf(fid, 'JAR    Add\n');
    fprintf(fid, 'GST    462.1\n');
    fprintf(fid, 'GSI    99.8\n');
    fprintf(fid, 'TE     0\n');
    fprintf(fid, 'HCF    512\n');
    fprintf(fid, 'HSW    99.8\n');
    fprintf(fid, 'NGA    -1\n');
    fprintf(fid, 'NOF    0\n');
    fprintf(fid, 'MF     0\n');
    fprintf(fid, 'MP     0\n');
    fprintf(fid, 'MCA    -1\n');
    fprintf(fid, 'RMA    0\n');
    fprintf(fid, 'RRG    0\n');
    fprintf(fid, 'RPH    0\n');
    fprintf(fid, 'ROF    0\n');
    fprintf(fid, 'RCT    0\n');
    fprintf(fid, 'RTC    0\n');
    fprintf(fid, 'RMF    0\n');
    fprintf(fid, 'RHA    1\n');
    fprintf(fid, 'RRE    1\n');
    fprintf(fid, 'RES    %i\n', length(src.y));
    fprintf(fid, 'DTM    1\n');
    fprintf(fid, 'DSD    20000\n');
    fprintf(fid, 'DCT    0\n');
    fprintf(fid, 'DTR    0\n');
    fprintf(fid, 'DCA    ON\n');
    fprintf(fid, 'DCB    OFF\n');
    fprintf(fid, 'DDM    OFF\n');
    fprintf(fid, 'DRS    4096\n');
    fprintf(fid, 'PPL    OFF\n');
    fprintf(fid, 'PFP    2\n');
    fprintf(fid, 'PSP    1\n');
    fprintf(fid, 'POF    0\n');
    fprintf(fid, 'PFR    ON\n');
    fprintf(fid, 'EMF    3352.1\n');
    fprintf(fid, 'ESF    20\n');
    fprintf(fid, 'ESW    10\n');
    fprintf(fid, 'EFD    99.77\n');
    fprintf(fid, 'EPF    10\n');
    fprintf(fid, 'ESP    20\n');
    fprintf(fid, 'EPP    63\n');
    fprintf(fid, 'EOP    0\n');
    fprintf(fid, 'EPH    0\n');
    fprintf(fid, 'FME    NONE\n');
    fprintf(fid, 'FWI    2\n');
    fprintf(fid, 'FOP    2\n');
    fprintf(fid, 'FER    2\n');
    fclose(fid);

    
end
