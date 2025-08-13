

function res = epr_spin_probe_calibration(entry)

OX63 = 'Ox63';
OX71 = 'Ox71';
LiPC = 'LiPC';
SOx71 = 'SOx71';


% 2mM pbs calibration at 37 11/16/2020
clb{1}.probe     = OX71; 
clb{end}.media     = '2mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1357; %[MS-1]
clb{end}.slopeT1     = 113.3798; %[torr/MS-1]
clb{end}.interceptT2 = 0.3597; %[MS-1]
clb{end}.slopeT2     = 109.3697; %[torr/MS-1]
clb{end}.date      = '20201611'; %date


% 2mM pbs calibration at 37 11/16/2020
clb{1}.probe     = OX71; 
clb{end}.media     = '2mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1357; %[MS-1]
clb{end}.slopeT1     = 113.3798; %[torr/MS-1]
clb{end}.interceptT2 = 0.3597; %[MS-1]
clb{end}.slopeT2     = 109.3697; %[torr/MS-1]
clb{end}.date      = '20201611'; %date


% 1mM pbs calibration at 37C 7/29/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1382; %[MS-1]
clb{end}.slopeT1     = 116.6973; %[torr/MS-1]
clb{end}.interceptT2 = 0.2759; %[MS-1]
clb{end}.slopeT2     = 120.1328; %[torr/MS-1]
clb{end}.date      = '20200729'; %date

% 1mM diH2O calibration at 24C 10/25/2022
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM diH2O WVa'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.181390; %[MS-1]
clb{end}.slopeT1     = 112.127; %[torr/MS-1]
clb{end}.interceptT2 = 0.238670; %[MS-1]
clb{end}.slopeT2     = 116.245; %[torr/MS-1]
clb{end}.date      = '20200729'; %date

% 1mM pbs calibration at 37C 12/01/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM PBS(New)'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1244; %[MS-1]
clb{end}.slopeT1     = 109.6640; %[torr/MS-1]
clb{end}.interceptT2 = 0.1975; %[MS-1]
clb{end}.slopeT2     = 93.4971; %[torr/MS-1]
clb{end}.date      = '20200729'; %date

% 0.5mM pbs calibration at 37 12/1/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0.5mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1279; %[MS-1]
clb{end}.slopeT1     = 113.2347; %[torr/MS-1]
clb{end}.interceptT2 = 0.1564; %[MS-1]
clb{end}.slopeT2     = 85.8440; %[torr/MS-1]
clb{end}.date      = '20200112'; %date

% 0.5mM pbs calibration at Room Temp 7/14/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0.5mM PBS'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1183; %[MS-1]
clb{end}.slopeT1     = 125.0484; %[torr/MS-1]
clb{end}.interceptT2 = 0.1501; %[MS-1]
clb{end}.slopeT2     = 121.9413; %[torr/MS-1]
clb{end}.date      = '20201407'; %date

% 0.25mM pbs calibrations at Room Temp 11/12/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0.25mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1021; %[MS-1]
clb{end}.slopeT1     = 130.4815; %[torr/MS-1]
clb{end}.interceptT2 = 0.1586; %[MS-1]
clb{end}.slopeT2     = 106.6073; %[torr/MS-1]
clb{end}.date      = '20201211'; %date

% 0.25mM pbs calibrations at Room Temp 6/10/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0.25mM PBS'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1148; %[MS-1]
clb{end}.slopeT1     = 126.0270; %[torr/MS-1]
clb{end}.interceptT2 = 0.1723; %[MS-1]
clb{end}.slopeT2     = 116.7172; %[torr/MS-1]
clb{end}.date      = '20201006'; %date

% 0.125 pbs calibrations at Room Temp 6/11/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0.125mM PBS'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1089; %[MS-1]
clb{end}.slopeT1     = 124.6470; %[torr/MS-1]
clb{end}.interceptT2 = 0.1384; %[MS-1]
clb{end}.slopeT2     = 107.4598; %[torr/MS-1]
clb{end}.date      = '20201106'; %date

% 8mM pbs calibration at 37 12/1/2020
clb{1}.probe     = OX71; 
clb{end}.media     = '8mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1666; %[MS-1]
clb{end}.slopeT1     = 114.3582; %[torr/MS-1]
clb{end}.interceptT2 = 0.5142; %[MS-1]
clb{end}.slopeT2     = 110.8468; %[torr/MS-1]
clb{end}.date      = '20200112'; %date

% 4mM pbs calibration at 37 11/30/2020
clb{1}.probe     = OX71; 
clb{end}.media     = '4mM PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1317; %[MS-1]
clb{end}.slopeT1     = 113.0584; %[torr/MS-1]
clb{end}.interceptT2 = 0.2351; %[MS-1]
clb{end}.slopeT2     = 100.5438; %[torr/MS-1]
clb{end}.date      = '20203011'; %date

% 1mM DiH2O calibration at Room Temp 7/17/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM DiH2O'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1076; %[MS-1]
clb{end}.slopeT1     = 118.1879; %[torr/MS-1]
clb{end}.interceptT2 = 0.167; %[MS-1]
clb{end}.slopeT2     = 119.6976; %[torr/MS-1]
clb{end}.date      = '20201707'; %date

% 0mM DiH2O calibration at Room Temp -/-/-
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0mM DiH2O'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1069; %[MS-1]
clb{end}.slopeT1     = 117.92; %[torr/MS-1]
clb{end}.interceptT2 = 0.1616; %[MS-1]
clb{end}.slopeT2     = 117.69; %[torr/MS-1]
clb{end}.date      = '20201707'; %date

% 0mM DiH2O calibration at 30C -/-/-
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0mM DiH2O'; 
clb{end}.T         = 30; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1135; %[MS-1]
clb{end}.slopeT1     = 113.06; %[torr/MS-1]
clb{end}.interceptT2 = 0.1692; %[MS-1]
clb{end}.slopeT2     = 117.24; %[torr/MS-1]
clb{end}.date      = '20201707'; %date

% 0mM DiH2O calibration at 37C -/-/-
clb{end+1}.probe     = OX71; 
clb{end}.media     = '0mM DiH2O'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1138; %[MS-1]
clb{end}.slopeT1     = 108.42; %[torr/MS-1]
clb{end}.interceptT2 = 0.1642; %[MS-1]
clb{end}.slopeT2     = 112.87; %[torr/MS-1]
clb{end}.date      = '20201707'; %date


% 5 mg LiPC calibration at 24C 11/04/2022
clb{end+1}.probe     = LiPC; 
clb{end}.media     = '5mg powder in 6x50mm tube'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 710; %MHz 
clb{end}.interceptT1 = 0.2540; %[MS-1] 31mG
clb{end}.slopeT1     = 12.188; %[torr/MS-1]
clb{end}.interceptT2 = 0.3730; %[MS-1]
clb{end}.slopeT2     = 13.880; %[torr/MS-1] 0.215 torr/mG
clb{end}.date      = '20221104'; %date

% 1mM in 1% Gelatin calibration at 24C 09/14/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM 1% Gelatin'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1412; %[MS-1]
clb{end}.slopeT1     = 135.16; %[torr/MS-1]
clb{end}.interceptT2 = 0.3830; %[MS-1]
clb{end}.slopeT2     = 125.51; %[torr/MS-1]
clb{end}.date      = '202014009'; %date

% 1mM in 1% Gelatin calibration at 24C 08/02/2021
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM 1% Gelatin Refrigerated'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.123746; %[MS-1]
clb{end}.slopeT1     = 125.110; %[torr/MS-1]
clb{end}.interceptT2 = 0.305097; %[MS-1]
clb{end}.slopeT2     = 97.884; %[torr/MS-1]
clb{end}.date      = '20210208'; %date


% 1mM in 1% Gelatin calibration at 30C 09/08/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM 1% Gelatin'; 
clb{end}.T         = 30; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1257; %[MS-1]
clb{end}.slopeT1     = 123.32; %[torr/MS-1]
clb{end}.interceptT2 = 0.3299; %[MS-1]
clb{end}.slopeT2     = 124.49; %[torr/MS-1]
clb{end}.date      = '20200809'; %date

% 1mM in 1% Gelatin calibration at 37C 09/04/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM 1% Gelatin'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1191; %[MS-1]
clb{end}.slopeT1     = 114.13; %[torr/MS-1]
clb{end}.interceptT2 = 0.3136; %[MS-1]
clb{end}.slopeT2     = 119.44; %[torr/MS-1]
clb{end}.date      = '20200409'; %date

% 2% Agarose 1mM OX071 calibration at 37 C 9/10-11/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM 2% Agarose'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1253; %[MS-1]
clb{end}.slopeT1     = 123.8246; %[torr/MS-1]
clb{end}.interceptT2 = 0.2683; %[MS-1]
clb{end}.slopeT2     = 126.2669; %[torr/MS-1]
clb{end}.date      = '20201109'; %date

% Cornell 2% Alginate 1mM OX071 calibration at 37 C 9/19 and 9/10/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM Cornell 2% Alginate'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1223; %[MS-1]
clb{end}.slopeT1     = 119.5819; %[torr/MS-1]
clb{end}.interceptT2 = 0.2599; %[MS-1]
clb{end}.slopeT2     = 114.7216; %[torr/MS-1]
clb{end}.date      = '20201009'; %date

% 154mM pbs calibration at 37C -/-/-
clb{end+1}.probe     = OX71; 
clb{end}.media     = '154mM PBS2'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1246; %[MS-1]
clb{end}.slopeT1     = 112.93; %[torr/MS-1]
clb{end}.interceptT2 = 0.2555; %[MS-1]
clb{end}.slopeT2     = 123.15; %[torr/MS-1]
clb{end}.date      = '20200902'; %date

% 154mM pbs calibration at 30C -/-/-
clb{end+1}.probe     = OX71; 
clb{end}.media     = '154mM PBS2'; 
clb{end}.T         = 30; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1302; %[MS-1]
clb{end}.slopeT1     = 124.58; %[torr/MS-1]
clb{end}.interceptT2 = 0.1692; %[MS-1]
clb{end}.slopeT2     = 117.239; %[torr/MS-1]
clb{end}.date      = '20200902'; %date


% 1mM pbs calibration at 24C 09/02/2020
clb{end+1}.probe     = OX71; 
clb{end}.media     = '1mM PBS2'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1316; %[MS-1]
clb{end}.slopeT1     = 131.85; %[torr/MS-1]
clb{end}.interceptT2 = 0.1062; %[MS-1]
clb{end}.slopeT2     = 100.4711; %[torr/MS-1]
clb{end}.date      = '20200902'; %date

% 1mM LiPc pbs calibration at 24C 09/02/2020
clb{end+1}.probe     = LiPC; 
clb{end}.media     = '--'; 
clb{end}.T         = 24; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.5458; %[MS-1] 31mG
clb{end}.slopeT1     = 14.0; %[torr/MS-1]
clb{end}.interceptT2 = 0.5458; %[MS-1]
clb{end}.slopeT2     = 14.0; %[torr/MS-1] 0.215 torr/mG
clb{end}.date      = '20200214'; %date

% 1mM LiPC pbs calibration at 37C 09/02/2020
clb{end+1}.probe     = LiPC; 
clb{end}.media     = '--'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.5458; %[MS-1] 31mG
clb{end}.slopeT1     = 15.4; %[torr/MS-1]
clb{end}.interceptT2 = 0.4458; %[MS-1]
clb{end}.slopeT2     = 15.7; %[torr/MS-1] 0.215 torr/mG
clb{end}.date      = '20200214'; %date

% LiPC pbs calibration at 37C 09/02/2020
clb{end+1}.probe     = LiPC; 
clb{end}.media     = '-mouse-'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.6013; %[MS-1] 31mG
clb{end}.slopeT1     = 23.2442; %[torr/MS-1]
clb{end}.interceptT2 = 0.4458; % ????[MS-1]
clb{end}.slopeT2     = 15.7; % ???? [torr/MS-1] 0.215 torr/mG
clb{end}.date      = '20210917'; %date

%Calibration_SOX71_1 mM in PBS at 37 C_2022-11-16
clb{end+1}.probe     = SOx71; 
clb{end}.media     = '1mM in PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1266; %[MS-1] 31mG
clb{end}.slopeT1     = 205.4831; %[torr/MS-1]
clb{end}.interceptT2 = 0.1281; % [MS-1]
clb{end}.slopeT2     = 243.0284; % [torr/MS-1] 
clb{end}.date      = '20231116'; %date

% 1mM ox071 in Complete DMEM Cell Culture Calibration at 37C 
clb{end+1}.probe     = OX71; 
clb{end}.media     = 'Complete DMEM'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1537; %[MS-1]
clb{end}.slopeT1     = 123.3941; %[torr/MS-1]
clb{end}.interceptT2 = .614; %[MS-1]
clb{end}.slopeT2     = 64.5495; %[torr/MS-1]
clb{end}.date      = '20220729'; %date


% 1mM ox071 in 66% VitroGel Calibration at 37C 
clb{end+1}.probe     = OX71; 
clb{end}.media     = '66% VitroGel'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.130; %[MS-1]
clb{end}.slopeT1     = 113.484; %[torr/MS-1]
clb{end}.interceptT2 = 0.140; %[MS-1]
clb{end}.slopeT2     = 124.207; %[torr/MS-1]
clb{end}.date      = '20220729'; %date

% Culture Media in 1M in Min6 Cell calibration at 37C 
clb{end+1}.probe     = OX71; 
clb{end}.media     = 'Culture Media Min6'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.1195; %[MS-1]
clb{end}.slopeT1     = 114.05; %[torr/MS-1]
clb{end}.interceptT2 = 0.2471; %[MS-1]
clb{end}.slopeT2     = 119.84; %[torr/MS-1]
clb{end}.date      = '20202307'; %date

% 1mM Trityl w/ 100umol/L Mn ion in PBS @37
clb{1}.probe     = OX71; 
clb{end}.media     = '1mM w/ 100uM/L Mn-ion in PBS'; 
clb{end}.T         = 37; 
clb{end}.Frequency = 720; %MHz 
clb{end}.interceptT1 = 0.392; %[MS-1]
clb{end}.slopeT1     = 112.581; %[torr/MS-1]
clb{end}.interceptT2 = 0.4758; %[MS-1]
clb{end}.slopeT2     = 90.3967; %[torr/MS-1]
clb{end}.date      = '20220614'; %date

if ~exist('entry', 'var')
  res = {};
  for ii=1:length(clb)
    res{end+1}=make_entry(clb{ii});
  end
  res = sort(res);
else
  for ii=1:length(clb)
    if strcmp(entry, make_entry(clb{ii})) == 1
      res = clb{ii};
      return;
    end
  end
end

function res = make_entry(clb)
res = '';

if isfield(clb, 'probe')
  res = [res , clb.probe];
end
if isfield(clb, 'Frequency')
  res = sprintf('%s-%3.0f', res, clb.Frequency);
end
if isfield(clb, 'media')
  res = [res , '-', clb.media];
end
if isfield(clb, 'T')
  res = sprintf('%s-%2.0fC', res, clb.T);
end
