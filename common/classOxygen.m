% --------------------------------------------------------------------
% this class standard units are [bar] for pressure and [K] for temperature
classdef classOxygen
  properties (Constant)
    % Units
    c1Atm  = 1.01325; % Atm in bar
    c1torr = 1/750.061685; % torr in bar
    c1psi = 0.06894757; % psi in bar
    % Volume ratio in dry air
    cDryAirO2Vol = 0.2095; % volume ratio of O2
    cDryAirN2Vol = 0.7809; % volume ratio of N2
    cDryAirArVol = 0.0093; % volume ratio of Ar
    cDryAirCO2Vol = 0.0004; % volume ratio of CO2
    % Molar masses
    cMO2   = 31.9988; % g/mol
    CMAir  = 28.9647; % average molar weight of air
    cAirO2Mol = 0.20946; % molar ratio [mol/mol_air]
    cMH2O   = 18.02; % g/mol
    cMNaCl = 58.44; % molar mass of NaCl
    cMNa = 22.989; % molar mass of Na
    cMK = 39.0983; % molar mass of K
    cMCl = 35.453; % molar mass of Cl
    cMSO4 = 96.06; % molar mass of SO4
    cMHPO4 = 95.982; % molar mass of HPO4
    % Phys. constants
    cTemp0C = 273.15; % Temperature of 0C in K
    cR = 0.0831446261815324e-3; % m3?bar?K?1?mol?1
    cNav = 6.0221409e+23; %mol?1
    cRT = 293.15; % Room temperature 20C
    % PBS
    cPBS_Na = 0.157; % M Na+ in PBS
    cPBS_Cl = 0.140; % M Cl- in PBS
    cPBS_K = 0.00445; % M K+ in PBS    
    cPBS_HPO4 = 0.00101; % M HPO4 2-  in PBS 
    cPBS_H2PO4 = 0.00176; % M H2PO4 -  in PBS
    cSalinityPBS = 0.009885; % g of salt per 1L
    cSeaWater_Na = 0.469; % M Na+
    cSeaWater_Cl = 0.546; % M Cl-
    cSeaWater_Mg = 0.0528; % M Mg+
    cSeaWater_K = 0.00102; % M K+    
    cSeaWater_SO4 = 0.00282; % mM S04 2-
    cSalinitySeaWater = 0.035; % g of salt per 1L
  end
  % --------------------------------------------------------------------
  methods (Static)
    function torr = bar2torr(bar)
      torr = bar / classOxygen.c1torr;
    end
    function bar = torr2bar(torr)
      bar = torr * classOxygen.c1torr;
    end
    % Temperature in C
    % humidity is fraction of 1
    function Vbar = PH2O(Tk, humidity)
      if ~exist('humidity', 'var'), humidity = 1; end
      % PT in torr, first element is 0C step 1C
      PT = [4.6, 4.9, 5.3, 5.7, 6.1, 6.5, 7, 7.5, 8.1, 8.6, 9.2, 9.8, 10.5, 11.2, 12, 12.8, 13.6, 14.5, 15.5, 16.5, ...
        17.5, 18.7, 19.8, 21.1, 22.4, 23.8, 25.2, 26.7, 28.4, 30, 31.8, 33.7, 35.7, 37.7, 39.9, 42.2, 44.6, 47.1, 49.7, 52.4, ...
        55.3, 58.3, 61.5, 64.8, 68.3, 71.9, 75.7, 79.6, 83.7, 88, 92.5];
      Vbar = classOxygen.torr2bar(interp1(0:length(PT)-1, PT, Tk-classOxygen.cTemp0C, 'nearest','extrap') * humidity);
    end
    function SolO2 = SolO2DOTABLES_molkg(Tk, Pbar, fugacity, humidity, salinity)
      % https://water.usgs.gov/water-resources/software/DOTABLES/
      % 760 ppm and normal fugacity
      DOTsalinity = [0,1,2,3,4,5,10,15,20,25,30,35,40];
      DOT = [10	11.29	11.22	11.14	11.07	11	10.93	10.59	10.26	9.93	9.62	9.32	9.02	8.74;...
      11	11.03	10.96	10.89	10.82	10.75	10.68	10.35	10.03	9.71	9.41	9.12	8.83	8.56;...
      12	10.78	10.71	10.64	10.58	10.51	10.44	10.12	9.81	9.5	9.21	8.92	8.65	8.38;...
      13	10.54	10.47	10.41	10.34	10.28	10.21	9.9	9.6	9.3	9.02	8.74	8.47	8.21;...
      14	10.31	10.24	10.18	10.12	10.05	9.99	9.69	9.39	9.11	8.83	8.56	8.3	8.05;...
      15	10.08	10.02	9.96	9.9	9.84	9.78	9.48	9.2	8.92	8.65	8.39	8.14	7.89;...
      16	9.87	9.81	9.75	9.69	9.63	9.57	9.29	9.01	8.74	8.48	8.22	7.98	7.74;...
      17	9.66	9.61	9.55	9.49	9.43	9.38	9.1	8.83	8.57	8.31	8.06	7.82	7.59;...
      18	9.47	9.41	9.35	9.3	9.24	9.19	8.92	8.65	8.4	8.15	7.91	7.68	7.45;...
      19	9.28	9.22	9.17	9.11	9.06	9	8.74	8.48	8.24	8	7.76	7.53	7.31;...
      20	9.09	9.04	8.99	8.93	8.88	8.83	8.57	8.32	8.08	7.85	7.62	7.4	7.18;...
      21	8.92	8.86	8.81	8.76	8.71	8.66	8.41	8.17	7.93	7.7	7.48	7.26	7.05;...
      22	8.74	8.69	8.64	8.59	8.54	8.49	8.25	8.01	7.78	7.56	7.34	7.13	6.93;...
      23	8.58	8.53	8.48	8.43	8.38	8.33	8.1	7.87	7.64	7.43	7.21	7.01	6.81;...
      24	8.42	8.37	8.32	8.27	8.23	8.18	7.95	7.73	7.51	7.3	7.09	6.89	6.69;...
      25	8.26	8.22	8.17	8.12	8.08	8.03	7.81	7.59	7.38	7.17	6.97	6.77	6.58;...
      26	8.11	8.07	8.02	7.98	7.93	7.89	7.67	7.45	7.25	7.05	6.85	6.66	6.47;...
      27	7.97	7.92	7.88	7.84	7.79	7.75	7.53	7.33	7.12	6.93	6.73	6.55	6.37;...
      28	7.83	7.78	7.74	7.7	7.66	7.61	7.4	7.2	7	6.81	6.62	6.44	6.26;...
      29	7.69	7.65	7.61	7.56	7.52	7.48	7.28	7.08	6.89	6.7	6.52	6.34	6.16;...
      30	7.56	7.52	7.48	7.44	7.39	7.35	7.15	6.96	6.77	6.59	6.41	6.24	6.07;...
      31	7.43	7.39	7.35	7.31	7.27	7.23	7.04	6.85	6.66	6.48	6.31	6.14	5.97;...
      32	7.3	7.27	7.23	7.19	7.15	7.11	6.92	6.73	6.55	6.38	6.21	6.04	5.88;...
      33	7.18	7.14	7.11	7.07	7.03	6.99	6.81	6.63	6.45	6.28	6.11	5.95	5.79;...
      34	7.06	7.03	6.99	6.95	6.92	6.88	6.7	6.52	6.35	6.18	6.02	5.86	5.7;...
      35	6.95	6.91	6.88	6.84	6.8	6.77	6.59	6.42	6.25	6.08	5.92	5.77	5.62;...
      36	6.84	6.8	6.76	6.73	6.69	6.66	6.48	6.32	6.15	5.99	5.83	5.68	5.53;...
      37	6.73	6.69	6.66	6.62	6.59	6.55	6.38	6.22	6.06	5.9	5.75	5.6	5.45;...
      38	6.62	6.59	6.55	6.52	6.48	6.45	6.28	6.12	5.96	5.81	5.66	5.51	5.37;...
      39	6.52	6.48	6.45	6.41	6.38	6.35	6.19	6.03	5.87	5.72	5.58	5.43	5.29;...
      40	6.41	6.38	6.35	6.31	6.28	6.25	6.09	5.93	5.78	5.64	5.49	5.35	5.22];
      SolO2 = Pbar/classOxygen.c1Atm * interp2(DOTsalinity, DOT(:,1)+classOxygen.cTemp0C, DOT(:,2:end), salinity, Tk) * fugacity/classOxygen.cDryAirO2Vol / classOxygen.cMO2*1e-3;
    end
    function SolO2 = SolO2seawater_molkg(Tk, Pbar, fugacity, humidity)
      %       https://www.engineeringtoolbox.com/oxygen-solubility-water-d_841.html
      SOL = [349, 308, 275, 248, 225, 206, 190, 176, 165, 154, 146]; % at 1 Atm and (assumed) correct fugacity and H2O 
      T = classOxygen.cTemp0C + (0:5:50);
      SolO2 = Pbar/classOxygen.c1Atm * interp1(classOxygen.cTemp0C+(0:5:50), SOL, Tk, 'nearest','extrap') * fugacity/classOxygen.cDryAirO2Vol*1e-6;
    end
    function pressureO2 = O2presssure(Pbar, fugacity, Tk, humidity)
      pressureO2 = (Pbar - humidity*classOxygen.PH2O(Tk,1))*fugacity;
    end
    function SolO2_molkg = SolO2GC_molkg(Tk, Pbar, fugacity, humidity, ions)
      % Geochimica et Cosmochimica Acta 74 (2010) 5631–5640
      c1 = 3.2832E1; c2 = -4.2731E-2; c3 = -4.4074E3;
      c4 = 1.5253E-5; c6 = 1.0279E-3; c7 = -2.9945e-3;
      c9 = 2.6179e-3; c11 = 5.6660E-5;
      ion_contribution = 0;
      lambda_Na = 1.9997e-1;
      eta_NaCl = -1.2793e-2;
      if ~exist('ions', 'var'), ions = []; end
      if isfield(ions, 'Na'), ion_contribution = ion_contribution - 2*lambda_Na*ions.Na; end
      if isfield(ions, 'Cl'), ion_contribution = ion_contribution - eta_NaCl*ions.Cl; end
      mu = c1 + c2*Tk + c3./Tk + c4*Tk.^2 + c6*Pbar + c7*log(Tk) + c9*Pbar./(630-Tk);
      SolO2_molkg = classOxygen.O2presssure(Pbar, fugacity, Tk, humidity) * exp(-mu + ion_contribution);
    end
    function Sol_molkg = SolO2H_molkg(Tk, Pbar, fugacity, humidity)
      kH_298 = 0.0013;
      Sol_molkg = classOxygen.O2presssure(Pbar, fugacity, Tk, humidity) * kH_298 * exp(1500*(1/Tk - 1/298.15));
    end
    function mol = MolGas(VolM3, Pbar, Tk)
      mol = Pbar * VolM3 / (classOxygen.cR * Tk); % PV = nRT
    end
    function test_sol(Tk, Pbar)
      clc
      fprintf('Solubility(T=%4.1fC, P=%4.1fatm) mol/kg\n', Tk-classOxygen.cTemp0C, Pbar/classOxygen.c1Atm);
      fprintf('---------------------------------------\n');
      disp('SolO2DOTABLES_molkg DI');
      disp(classOxygen.SolO2DOTABLES_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0, 0*1000) * 1000)
      disp('SolO2DOTABLES_molkg PBS (~0.01)');
      disp(classOxygen.SolO2DOTABLES_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0, classOxygen.cSalinityPBS*1000) * 1000)
      disp('SolO2DOTABLES_molkg seawater (0.035)');
      disp(classOxygen.SolO2DOTABLES_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0, classOxygen.cSalinitySeaWater*1000) * 1000)
      disp('SolO2H_molkg')
      disp(classOxygen.SolO2H_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0) * 1000)
      disp('SolO2GC_molkg')
      disp(classOxygen.SolO2GC_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0) * 1000)
      disp('SolO2GC_molkg SeaWater ')
      ions = struct('Na', classOxygen.cSeaWater_Na, 'Cl', classOxygen.cSeaWater_Cl);
      disp(classOxygen.SolO2GC_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0, ions) * 1000)
      disp('SolO2seawater_molkg')
      disp(classOxygen.SolO2seawater_molkg(Tk, Pbar, classOxygen.cDryAirO2Vol, 0) * 1000)
    end
  end
end
