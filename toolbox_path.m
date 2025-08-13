% this script sets up the Imaging paths and the local work path
% Feb. 16 , 2004 CH
% mod. 9-18-06 to automate, no user editing required
% Type path in the command window and copy any custom paths 
% modify this file to add your custom paths
%
% statements using "-begin" add to the beginning of the path and  
% statements using "-end" append the path  
% commented lines are not being used
 
fixeprPath = questdlg('Reset EPR/UC path');
if strcmp(fixeprPath,'Yes')
  restoredefaultpath;  
end
 epr_toolbox_path = 'C:\Users\kmeng\Documents\EPRI-T';
%   epr_toolbox_path = 'D:\CenterMATLAB';
%  epr_toolbox_path = 'c:\Users\admin\Dropbox\MATLAB';
%  epr_toolbox_path = '/Users/borisepel/Dropbox/MATLAB';
  
  addpath([epr_toolbox_path, filesep, 'common'], '-begin')

  % Calibration files for EPR imagers 
  %   addpath([epr_toolbox_path, filesep, 'Lab_Hardware _Calibration'], '-end')
    
  % MRI loader
  addpath([epr_toolbox_path, filesep, 'MRIloader'], '-end')
  
  % Report generators
  addpath([epr_toolbox_path, filesep, 'Reports'], '-begin')
  
  %   % Transformations library
  %   addpath([epr_toolbox_path, filesep, 'chuck', filesep, 'hmats'], '-begin')
  %   addpath([epr_toolbox_path, filesep, 'chuck', filesep, 'other'], '-begin')
  
  % Samples
  addpath([epr_toolbox_path, filesep, 'samples'], '-begin')

  % Common utils
  %   addpath([epr_toolbox_path, filesep, 'Utils'], '-begin')
  
  % MySQL database
  %   addpath([epr_toolbox_path, filesep, 'MySQL'], '-begin')
  
  % Numerical differentiation toolbox
  % http://www.mathworks.co.uk/matlabcentral/newsreader/view_thread/157530
  % http://www.mathworks.com/matlabcentral/fileexchange/13490
  %   addpath('c:\MATLAB\toolbox\DERIVESTsuite', '-end')
  
  % Forward projection
  addpath([epr_toolbox_path, filesep, 'radon'], '-begin')
  
  % FBP image reconstruction 32 and 64 bit
  % general FBP routines and MSPS index files
  addpath([epr_toolbox_path, filesep, 'iradon'], '-begin')
  addpath([epr_toolbox_path, filesep, 'MSPS'], '-begin')

  % multy-stage reconstruction
  addpath([epr_toolbox_path, filesep, 'iradon_mstage'], '-begin')
  % Jonathan Brian CPU/GPU
  addpath([epr_toolbox_path, filesep, 'iradon_Bryant3D'], '-begin')
  % Zhiwei's single stage CPU/GPU
  addpath([epr_toolbox_path, filesep, 'iradon_QiaoFBP3D'], '-begin')
  % Mark 3D+FT reconstruction
  addpath([epr_toolbox_path, filesep, 'iradon_Tseylin4D'], '-begin')
  
  % UIUC
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'iradon_UIUC'], '-begin')
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'iradon_UIUC', filesep, 'gridding_ls_3D'], '-begin')
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'iradon_UIUC', filesep, 'hftrec'], '-begin')
  
  % EPR imaging
  addpath([epr_toolbox_path, filesep, 'epri'], '-begin')
  addpath([epr_toolbox_path, filesep, 'epri', filesep, 'Scenario'], '-begin')
  addpath([epr_toolbox_path, filesep, 'eprfit'], '-begin')
%   addpath([epr_toolbox_path, filesep, 'Loaders'], '-begin')
  addpath([epr_toolbox_path, filesep, 'pviewer'], '-begin')
  
  %   % CW fit
  addpath([epr_toolbox_path, filesep, 'cwfit'], '-begin')
  %   addpath([epr_toolbox_path, filesep, 'colin_CW _R14'], '-begin')
  
  % Rapid Scan 
  addpath([epr_toolbox_path, filesep, 'rapid_scan'], '-begin')
  
  % Radiation treatment
  addpath([epr_toolbox_path, filesep, 'Treatment'], '-begin')

  % Di-nitroxide processing
  addpath([epr_toolbox_path, filesep, 'PxSSPx'], '-begin')
  
  % addpath Z:\Matlab\fwdProjection -begin
  % addpath Z:\Matlab\Half_Pi -begin
  % addpath Z:\Matlab\EPRI_Utility_Files -begin
  % addpath Z:\Matlab\chuck\fiducials -begin
  % addpath Z:\Matlab\tumor_roi -begin
  
  % addpath Z:\Matlab\dce_mri -end
  % Z:\Matlab\Imaging_utils is where misc programs are shared from CP &
  % CH
  % addpath Z:\Matlab\Imaging_utils -end
  % addpath Z:\Matlab\Oxycal -end
  % this is the work directory on 128.135.32.196
  % addpath Z:\Matlab\ -end
  
  % Data browser
  addpath([epr_toolbox_path, filesep, 'kazan'], '-end')
  addpath([epr_toolbox_path, filesep, 'kazan3'], '-end')
  % Volume visualization
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'sliceomatic2'], '-end')
  % EPR simulations
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'easyspin'], '-end')
  % Various staff
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'misc'], '-end')
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'FromChuck'], '-end')
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'Dosimetry_functions'], '-end')
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'restore_idl'], '-begin')
  % Registration software
  addpath([epr_toolbox_path, filesep, 'Arbuz2.0'], '-begin')
  % Image visualization software
  addpath([epr_toolbox_path, filesep, 'ibGUI'], '-begin')
  % Data Analysis
  addpath(['Z:\CenterDataProcessing'], '-begin')

  % Matlab bridge to Slicer3D
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'MatlabBridge'], '-end')
  
  % Simple addition gui for IMRT
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'AdditionGUI'], '-end')
  % Scripts to Export data in matlab structure vars
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'Matlab_export'], '-end')
  % Simple  gui for IMRT_ 3 CT thresholding
  addpath([epr_toolbox_path, filesep, '3dparty', filesep, 'FindskinGUI'], '-end')
  % Treatment functions
  addpath([epr_toolbox_path, filesep, 'Treatment'], '-end')
  % functions for plug planning
   addpath([epr_toolbox_path, filesep, 'IrradiationPlanning'], '-end')

disp('Loaded all epr_paths')



