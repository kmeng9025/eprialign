function SUV = imrt_PET2SUV(PET, parameters)

PreparationTime = safeget(parameters, 'PreparationTime', datetime("now")); 
InjectionTime = safeget(parameters, 'InjectionTime', datetime("now")); 
PostInjectionTime = safeget(parameters, 'PostInjectionTime', datetime("now")); 

AcquisitionStartTime = safeget(parameters, 'AcquisitionStartTime', datetime("now"));
AcquisitionEndTime = safeget(parameters, 'AcquisitionEndTime', datetime("now"));

SyringeDose = safeget(parameters, 'SyringeDose', 0); % activity in syringe
LeftoverDose = safeget(parameters, 'LeftoverDose', 0); % Lefover activity post injection
HalfLife = safeget(parameters, 'HalfLife', 109.771); % half live 19F minutes

InjectedDose = SyringeDose - LeftoverDose; 

MeasuredDose = InjectedDose * exp((-0.693 * minutes(AcquisitionStartTime-InjectionTime)) / HalfLife) ;

ImageResolution = safeget(parameters, 'ImageResolution', 0.05); % resolutin in mm
VoxelVolume = ImageResolution^3;

CF = safeget(parameters, 'CF', 0.0104);       % New conversion factor (2021) from Aaron ?
DAQ_1M = safeget(parameters, 'DAQ_1M', 23.7); % minutes to obtain CF with 1million counts
Weight = safeget(parameters, 'Weight', 25);   % minutes to obtain CF with 1million counts

uCi = PET*CF*DAQ_1M / minutes(AcquisitionEndTime-AcquisitionStartTime); % PET in uCi

SUV = uCi/VoxelVolume/(MeasuredDose/Weight);
