function raw_info = epr_ReadCWImageHeader(fid, opt)

% Check the file type
if strcmp(fgetl(fid),'% LabView Data')
  raw_info.data.Origin = 'LabView';
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Labview %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Load Labview header
  raw_info.dsc.path = fgetl(fid);
  raw_info.dsc.script = fgetl(fid);
  raw_info.pars_out(5) = fscanf(fid,'%d'); fgetl(fid); % N of Polar angles
  raw_info.pars_out(6) = fscanf(fid,'%d'); fgetl(fid); % N of Azimuthal angles
  raw_info.pars_out(7) = fscanf(fid,'%d'); fgetl(fid); % N of Acquired Spectral angles
  raw_info.pars_out(9) = fscanf(fid,'%d'); fgetl(fid); % N of points
  raw_info.pars_out(10) = fscanf(fid,'%d'); fgetl(fid); % Min. N of Scans
  %   raw_info.pars_out(11)  = fscanf(s{9},'%d'); % Voltage divider - obsolete
  fgetl(fid); raw_info.pars_out(11) = 9;
  raw_info.pars_out(12) = fscanf(fid,'%f'); fgetl(fid); % Dwell time
  raw_info.pars_out(18) = fscanf(fid,'%f'); fgetl(fid); % B center in gauss
  deltaH_G = fscanf(fid,'%f'); fgetl(fid); % spectral FOV in gauss
  raw_info.deltaH = deltaH_G*sqrt(2);
  raw_info.pars_out(19) = deltaH_G*sqrt(2);
  raw_info.pars_out(14) = fscanf(fid,'%f'); fgetl(fid); % RF Power in dBm
  raw_info.pars_out(13) = fscanf(fid,'%f'); fgetl(fid); % RF Freq. in MHz
  raw_info.pars_out(17) = fscanf(fid,'%f'); fgetl(fid); % B mod freq. in kHz
  raw_info.pars_out(16) = fscanf(fid,'%f'); fgetl(fid); % B mod amp in peak volts
  raw_info.pars_out(21) = fscanf(fid,'%f'); fgetl(fid); % Lock-in sensitivity in mV
  deltaL_cm = fscanf(fid,'%f'); fgetl(fid); % spatial FOV in cm
  raw_info.pars_out(8)=deltaL_cm*sqrt(2);
  current_center = fscanf(fid,'%f'); fgetl(fid); % current required to set B0 center (A)
  raw_info.pars_out(22) = fscanf(fid,'%d'); fgetl(fid); % 1: Full spectral Acq, 0: Half pi
  raw_info.pars_out(20) = fscanf(fid,'%f'); fgetl(fid); % Lock-in TimeConst in sec
  LPfilterRolloff = fscanf(fid,'%f');  fgetl(fid); % Lock-in LPfilterRolloff
  % 1=xyzb,2=xzb,3=yzb,4=xyb,5=xb,6=yb,7=zb
  raw_info.pars_out(23) = fscanf(fid,'%d'); fgetl(fid); % Image type
  raw_info.pars_out(25) = fscanf(fid,'%d'); fgetl(fid); % N of Total Spectral angles
  raw_info.data.FBP.swDefCode = 0; % swDefCode to be implemented later
  raw_info.pars_out(26) = 3; %swDefCode;
  % modify the protocol of scanMethod later
  [readValue, readCnt]=fscanf(fid,'%d'); %1: unif solid, 0: unif. linear
  if readCnt==1,  raw_info.pars_out(24)=9; fgetl(fid); else  raw_info.pars_out(24)=1; end
  raw_info.data.FBP.scanMethod = raw_info.pars_out(24);
  
  % new format of pars_out
  raw_info.format = 1;
  
  raw_info.data.FBP.imtype = raw_info.pars_out(23);
  raw_info.data.FBP.nSpec  = raw_info.pars_out(7);
  raw_info.data.FBP.nAz    = raw_info.pars_out(6);
  raw_info.data.FBP.nPolar = raw_info.pars_out(5);
else
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Modula %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  raw_info.data.Origin = 'Modula';

  fgetl(fid);
  raw_info.pars_out(5) = fscanf(fid,'Number of spatial polar angles:%d'); fgetl(fid);
  raw_info.pars_out(6) = fscanf(fid,'Number of spatial axi. angles:%d'); fgetl(fid);
  raw_info.pars_out(7) = fscanf(fid,'Number of acquired spectral angles:%d'); fgetl(fid);
  raw_info.pars_out(9) = fscanf(fid,'Number of points:%d'); fgetl(fid);
  raw_info.pars_out(10) = fscanf(fid,'Number of scans:%d'); fgetl(fid);
  raw_info.pars_out(11)  = fscanf(fid,'Voltage divider No.:%d'); fgetl(fid); % Voltage divider number VDN  % added 6/27/2002
  vdn = raw_info.pars_out(11); % Voltage divider number VDN for display in recon-gui
  raw_info.pars_out(12) = fscanf(fid,'Dwell time: %f'); fgetl(fid);
  raw_info.pars_out(18) = fscanf(fid,'B center:%f'); fgetl(fid); %swu
  deltaH_G = fscanf(fid,'Image B width (assuming VD#8) :%f'); fgetl(fid); % gauss - gets data from file set value*sqrt(2)
  
  raw_info.pars_out(14) = fscanf(fid,'RF Power :%f'); fgetl(fid);                  %dBm
  raw_info.pars_out(13) = fscanf(fid,'RF frequency :%f'); fgetl(fid);           %MHz
  raw_info.pars_out(17) = fscanf(fid,'B mod frequency:%f'); fgetl(fid);        %kHz
  raw_info.pars_out(16) = fscanf(fid,'B mod amplitude :%f'); fgetl(fid);        % gauss
  raw_info.pars_out(21) = fscanf(fid,'Sensitivity :%f'); fgetl(fid);                 % lockin gain mV
  deltaL_cm = fscanf(fid,'Length of the image :%f'); fgetl(fid);    %cm
  current_center = fscanf(fid,'Current at the center:%f'); fgetl(fid);
  raw_info.pars_out(22) = fscanf(fid,'Full Spectral acquisition (0/1):%d'); fgetl(fid);
  raw_info.pars_out(20) = fscanf(fid,'Lock in TConst:%f'); fgetl(fid);
  % 1=xyzb,2=xzb,3=yzb,4=xyb,5=xb,6=yb,7=zb
  raw_info.pars_out(23) = fscanf(fid,'Image type:%d'); fgetl(fid);
  raw_info.pars_out(25) = fscanf(fid,'Number of spectral angles over 180:%d'); fgetl(fid);
  
  [readValue, readCnt]=fscanf(fid,'Scan method:%d');
  if readCnt, raw_info.data.FBP.scanMethod=readValue;  fgetl(fid); else raw_info.data.FBP.scanMethod=1; end
  raw_info.pars_out(24) = raw_info.data.FBP.scanMethod;
  
  %
  raw_info.data.FBP.imtype = raw_info.pars_out(23);
  raw_info.data.FBP.nSpec  = raw_info.pars_out(7);
  raw_info.data.FBP.nAz    = raw_info.pars_out(6);
  raw_info.data.FBP.nPolar = raw_info.pars_out(5);
  
  jj=24;
  [readValue, readCnt]=fscanf(fid,'Coord. Pole:%d');jj=jj+readCnt;
  if readCnt, fgetl(fid); end
  [readValue, readCnt] = fscanf(fid,'SW def:%d');jj=jj+readCnt; fgetl(fid);
  if readCnt, raw_info.data.FBP.swDefCode=readValue; fgetl(fid); else raw_info.data.FBP.swDefCode=0; end
  raw_info.pars_out(26) = raw_info.data.FBP.swDefCode;
  if raw_info.pars_out(26)==3 % swDefCode
    % swDefCode=0: SW=sqrt(2)*deltaH/cos(alpha)
    % swDefCode=1: SW=deltaH*(1+tan(alpha))
    % swDefCode=2: SW=deltaH*(1+k*tan(alpha)), k=ROI/FOV
    % swDefCode=3: SW=deltaH*(1+k*tan(alpha)), k varies for grad. direction
    fprintf('Using: ');which('ROItable.txt')
  end
  % image header stores delta L x sqrt(2), delta H x sqrt(2) previously.
  % pars_out(19)=delta H x sqrt(2) (required for fitting process)
  % pars_out(8)=delta L x sqrt(2) (doesn't matter for fitting)
  if jj>25 % new format: deltaL, deltaH read from header
    raw_info.format = 1;
    raw_info.pars_out(19)=deltaH_G*sqrt(2);
    raw_info.pars_out(8)=deltaL_cm*sqrt(2);
  else % old format: deltaLsqrt2, deltaHsqrt2 read from header
    raw_info.format = 0;
    raw_info.pars_out(19)=deltaH_G; deltaH_G=deltaH_G/sqrt(2);
    raw_info.pars_out(8)=deltaL_cm; deltaL_cm=deltaL_cm/sqrt(2);
  end
  
  % --- for record only, not used in the reconstruction---------------
  [readValue, readCnt] = fscanf(fid,'Resonator Diameter (mm):%d'); jj=jj+readCnt;
  if readCnt, fgetl(fid); end
  [readValue, readCnt] = fscanf(fid,'ROI (mm):%d'); jj=jj+readCnt;
  if readCnt, fgetl(fid); end
  
  if raw_info.data.FBP.swDefCode==3
    %     fid=fopen('ROItable.txt','r');
    %     proj_index = fgetl(fid);
    %     while( strncmp(proj_index,'  1,',4)==0)
    %       proj_index = fgetl(fid);
    %     end
    %     while proj_index ~= -1
    %       % A(1): polar, A(2): azimuthal, A(3): fL1
    %       % B(1): polar, B(2): azimuthal, B(3): fL2
    %       A=fscanf(proj_index,'%d,%d: %f'); proj_index = fgetl(fid);
    %       B=fscanf(proj_index,'%d,%d: %f'); proj_index = fgetl(fid);
    %       fL1(A(1),A(2))=A(3); fL2(B(1),B(2))=B(3);
    %     end
  end
end

raw_info.data.FBP.angle_sampling = 'uniform_spatial_flip';
raw_info.ModFrequency = raw_info.pars_out(17)*1E-3/2.802;
raw_info.data.Modality = 'CWFBP';

raw_info.LockInTimeConst = raw_info.pars_out(20);
raw_info.DwellTime       = raw_info.pars_out(12);
raw_info.data.FBP.Npt = raw_info.pars_out(9);

% Sensitivity correction
% Mod amp correction
raw_info.ModAmplitude=raw_info.pars_out(16)*safeget(opt, 'mod_cal', 1);
ModBroadn_0 = safeget(opt, 'ModBroadn_0', 1);
ModBroadn_1 = safeget(opt, 'ModBroadn_1', 1);
raw_info.ampHH = raw_info.pars_out(21) * ...
 1 / (1 + ModBroadn_1 * (raw_info.ModAmplitude-ModBroadn_0)/ModBroadn_0);