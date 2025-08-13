% KAZAN dataviewer with plugins By Boris Epel & Alexey Silakov
% MPI of Bioinorganic Chemistry, Muelhaim an der Ruhr, 2003-2006
% University of Chicago, 2006
% Free for non-commercial use. Use the program at your own risk.
% The authors retain all rights.
% Contact: bepel@uchicago.edu, aus40@psu.edu

% boep 30-aug-06
% boep 10/27/2022

function varargout = kazan(varargin)

% parameters of main form app
% 'ax.x' column of data x values
% 'ax.xlabel' data x axis label
% 'ax.xunit' data x axis unit
% 'ax.y' column of data y values
% 'ax.ylabel' data y axis label
% 'ax.yunit' data y axis unit
% 'y' columns of data y values
% 'rx' column of result x values
% 'ry' columns of result y values
% 'rxlabel' result x axis unit
% 'script' final script that toolbox executes
% 'lscript' loading script

if nargin == 0  % LAUNCH GUI
  oldfig = getappdata(0, 'Kazan_Viewer_open');
  oldfig = oldfig(ishandle(oldfig));
  
  inifilename = fullfile(userpath, 'kazan.ini');
  ini = inimanage(inifilename);
  KazanViewer = safeget(ini, 'KazanViewer', []);
  try opc = str2double(safeget(KazanViewer, 'MultyKazan', '3')); catch err, ShowError(err); opc = 1; end
  if opc == 1 && ~isempty(oldfig), set(oldfig,'Visible','on'); return; end
  if isempty(oldfig), opc = 3; end % In order to get plugins
  
  fig = openfig(mfilename,'new');
  setappdata(0, 'Kazan_Viewer_open', [oldfig, fig]);
  
  check_figure_size(fig);
  
  % Generate a structure of app to pass to callbacks, and store it.
  app = guihandles(fig);
  
  % connect local version of function
  if exist('kv_ParameterReader_local', 'file') == 2
    app.ParameterReader = @kv_ParameterReader_local;
    disp(['Using local: ', which('kv_ParameterReader_local')]);
  else
    app.ParameterReader = @kv_ParameterReader;
  end
  if exist('kv_ScriptLoad_local', 'file') == 2
    app.ScriptLoad = @kv_ScriptLoadNew_local;
    disp(['Using local: ', which('kv_ScriptLoad_local')]);
  else
    app.ScriptLoad = @kv_ScriptLoadNew;
  end
  
  app.inifile = fullfile(userpath, 'kazan.ini');
  app.ini = inimanage(app.inifile);
  app.ini.KazanViewer = safeget(app.ini, 'KazanViewer', []);
  app.ini.KazanViewer.plugin_path = safeget(app.ini.KazanViewer, 'plugin_path', fileparts(which('kazan')));
  
  % Initialize plugins structure
  app.Plugins = {};
  % Load list of favorite directories
  for ii=1:5
    app.FavoriteDirectories{ii} = safeget(app.ini.KazanViewer, ['Favorite',num2str(ii)], '');
    app.ini.KazanViewer.(['Favorite',num2str(ii)]) = app.FavoriteDirectories{ii};
  end
  
  try app.autoZoomOff = safeget(app.ini.KazanViewer, 'autoZoomOff', 'off'); catch err, ShowError(err); app.autoZoomOff = 'off'; end
  try app.FixedZoomOpt = safeget(app.ini.KazanViewer, 'FixedZoomOpt', 'off'); catch err, ShowError(err); app.FixedZoomOpt = 'off'; end
  
  app.DefaultFont = 8;
  
  app.Message = '';
  app.MessageBlocker = tic;
  
  %**********************************************************************
  % Main Menu
  %**********************************************************************
  % menu File
  app.mFile = uimenu(app.MainFigure, 'Label','File', 'Tag', 'mFile');
  app.mFileOpen = uimenu(app.mFile, 'Label','Open Directory', 'Tag', 'mFileOpen', ...
    'Callback', 'kazan(''mFileSave'',gcbo,[],guidata(gcbo))');
  app.mFavoriteDirectoriesMain = uimenu(app.mFile, 'Label','Favorite Directories', 'Tag', 'mFavoriteDirectoriesMain');
  app.mFileSave = uimenu(app.mFile, 'Label','Save Source', 'Tag', 'mFileSave', ...
    'Separator', 'on', ...
    'Callback', 'kazan(''mFileSave'',gcbo,[],guidata(gcbo))');
  app.mFileSaveProc = uimenu(app.mFile, 'Label','Save Source with Processing', 'Tag', 'mFileSaveProc', ...
    'Callback', 'kazan(''mFileSave'',gcbo,[],guidata(gcbo))');
  app.mFileConvert = uimenu(app.mFile, 'Label','Convert directory to ASCII', 'Tag', 'mFileConvert', ...
    'Callback', 'kazan(''mFileSave'',gcbo,[],guidata(gcbo))');
  app.mFileSaveWorksp = uimenu(app.mFile, 'Label','Export to Workspace', 'Tag', 'mFileSaveWorksp', ...
    'Separator', 'on', 'Callback', 'kazan(''mFileWorkspace'',gcbo,[],guidata(gcbo))');
  app.mFileLoadWorksp = uimenu(app.mFile, 'Label','Load from Workspace', 'Tag', 'mFileLoadWorksp', ...
    'Callback', 'kazan(''mFileWorkspace'',gcbo,[],guidata(gcbo))');
  app.mFileAddHistory = uimenu(app.mFile, 'Label','Add to Selected', 'Tag', 'mFileAddHistory', ...
    'Separator', 'on', 'Callback', 'kazan(''mFileHistory'',gcbo,[],guidata(gcbo))');
  app.mFileRemoveHistory = uimenu(app.mFile, 'Label','Remove from Selected', 'Tag', 'mFileRemoveHistory', ...
    'Callback', 'kazan(''mFileHistory'',gcbo,[],guidata(gcbo))');
  app.mFileRemoveAllHistory = uimenu(app.mFile, 'Label','Clear Selected files', 'Tag', 'mFileRemoveAllHistory', ...
    'Callback', 'kazan(''mFileHistory'',gcbo,[],guidata(gcbo))');
  app.mFilePreviousHistory = uimenu(app.mFile, 'Label','Previous file', 'Tag', 'mFilePreviousHistory', ...
    'Accelerator', 'P', ...
    'Callback', 'kazan(''mFileHistory'',gcbo,[],guidata(gcbo))');
  app.mFileNextHistory = uimenu(app.mFile, 'Label','Next file', 'Tag', 'mFileNextHistory', ...
    'Accelerator', 'N', ...
    'Callback', 'kazan(''mFileHistory'',gcbo,[],guidata(gcbo))');
  
  % menu View
  app.mView = uimenu(app.MainFigure, 'Label','View', 'Tag', 'mView');
  app.mViewFont = uimenu(app.mView, 'Label','Font', 'Tag', 'mViewFont');
  app.mViewFont6 = uimenu(app.mViewFont, 'Label','-2', 'Tag', 'mViewFont6', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 6);
  app.mViewFont7 = uimenu(app.mViewFont, 'Label','-1', 'Tag', 'mViewFont7', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 7);
  app.mViewFont8 = uimenu(app.mViewFont, 'Label','Normal', 'Tag', 'mViewFont8', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 8,...
    'Checked', 'on');
  app.mViewFont9 = uimenu(app.mViewFont, 'Label','+1', 'Tag', 'mViewFont9', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 9);
  app.mViewFont10 = uimenu(app.mViewFont, 'Label','+2', 'Tag', 'mViewFont10', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 10);
  app.mViewFont11 = uimenu(app.mViewFont, 'Label','+3', 'Tag', 'mViewFont11', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 11);
  app.mViewFont12 = uimenu(app.mViewFont, 'Label','+4', 'Tag', 'mViewFont12', 'Callback', 'kazan(''ViewFont_Callback'',gcbo,guidata(gcbo))', 'UserData', 12);
  app.mViewSource = uimenu(app.mView, 'Label','Source', 'Tag', 'mViewSource', 'Separator', 'on', ...
    'Checked', 'on', 'Callback', 'kazan(''DataSource_Callback'',gcbo,[],guidata(gcbo))');
  app.mViewOutput = uimenu(app.mView, 'Label','Output', 'Tag', 'mViewOutput', ...
    'Callback', 'kazan(''DataSource_Callback'',gcbo,[],guidata(gcbo))');
  app.mViewSnO = uimenu(app.mView, 'Label','Source and Output', 'Tag', 'mViewSnO', ...
    'Callback', 'kazan(''DataSource_Callback'',gcbo,[],guidata(gcbo))');
  app.mViewReal = uimenu(app.mView, 'Label','Show real part', 'Tag', 'mViewReal', ...
    'Separator', 'on', 'Checked', 'on', 'Accelerator', 'R', ...
    'Callback', 'kazan(''DataType_Callback'',gcbo,[],guidata(gcbo))');
  app.mViewImag = uimenu(app.mView, 'Label','Show imaginary part', 'Tag', 'mViewImag', ...
    'Accelerator', 'I', 'Callback', 'kazan(''DataType_Callback'',gcbo,[],guidata(gcbo))');
  app.mViewMagnitude = uimenu(app.mView, 'Label','Show magnitude', 'Tag', 'mViewMagnitude', ...
    'Accelerator', 'M', 'Callback', 'kazan(''DataType_Callback'',gcbo,[],guidata(gcbo))');
  app.mViewAxistight = uimenu(app.mView, 'Label','Axis tight', 'Tag', 'mViewAxistight', ...
    'Checked', 'on', 'Separator', 'on', 'Accelerator', 'T',...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewAxisauto = uimenu(app.mView, 'Label','Axis auto', 'Tag', 'mViewAxisauto', ...
    'Accelerator', 'A', ...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewAxisfixed = uimenu(app.mView, 'Label','Axis fixed', 'Tag', 'mViewAxisfixed', ...
    'Accelerator', 'F', ...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewShowGrid = uimenu(app.mView, 'Label','Show Grid', 'Tag', 'mViewShowGrid', ...
    'Separator', 'on', 'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewShowGridMinor = uimenu(app.mView, 'Label','Show Minor Grid', 'Tag', 'mViewShowGrid', ...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewAxisnormal = uimenu(app.mView, 'Label','Aspect normaL', 'Tag', 'mViewAxisnormal', ...
    'Separator', 'on',  'Accelerator', 'L', ...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewAxissquare = uimenu(app.mView, 'Label','Aspect square', 'Tag', 'mViewAxissquare', ...
    'Accelerator', 'S', ...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewAxisEqual = uimenu(app.mView, 'Label','Axis equal', 'Tag', 'mViewAxisEqual', ...
    'Accelerator', 'Q', ...
    'Callback', 'kazan(''mViewAxisSet'',gcbo,[],guidata(gcbo))');
  app.mViewRange = uimenu(app.mView, 'Label','Select View Area', 'Tag', 'mViewRange', ...
    'Separator', 'on', 'Accelerator', 'V', ...
    'Callback', 'kazan(''mViewRange'',gcbo,[],guidata(gcbo))');
  % menu Tools
  app.mTools = uimenu(app.MainFigure, 'Label','Tools', 'Tag', 'mTools');
  app.mToolsClearSource = uimenu(app.mTools, 'Label','Clear Source', 'Tag', 'mToolsClearSource', ...
    'Callback', 'kazan(''mTools'',gcbo,111,guidata(gcbo))');
  app.mToolsFixProcessing = uimenu(app.mTools, 'Label','Fix Source Processing', 'Tag', 'mToolsFixProcessing', ...
    'Callback', 'kazan(''mTools'',gcbo,111,guidata(gcbo))');
  app.mToolsFixSlice = uimenu(app.mTools, 'Label','Cut Slice', 'Tag', 'mToolsFixSlice', ...
    'Callback', 'kazan(''mTools'',gcbo,111,guidata(gcbo))');
  app.mToolsLoadscript = uimenu(app.mTools, 'Label','Show Data Load script', 'Tag', 'mToolsLoadscript', ...
    'Separator', 'on', 'Callback', 'kazan(''mTools'',gcbo,111,guidata(gcbo))');
  app.mToolsExecLoadscript = uimenu(app.mTools, 'Label','Execute Data Load Script', 'Tag', 'mToolsExecLoadscript', ...
    'Callback', 'kazan(''mTools'',gcbo,111,guidata(gcbo))');
  app.mToolsZoom = uimenu(app.mTools, 'Label','Zoom', 'Tag', 'mToolsZoom', 'Separator', 'on', ...
    'Callback', 'kazan(''mTools'',gcbo,[],guidata(gcbo))', ...
    'Accelerator', 'Z');
  app.mToolsAutoZoomOff = uimenu(app.mTools, 'Label','One Zoom per Time', 'Tag', 'mToolsAutoZoomOff', ...
    'Callback', 'kazan(''mTools'',gcbo,[],guidata(gcbo))', ...
    'Checked', app.autoZoomOff);
  app.mToolsZoomFixed = uimenu(app.mTools, 'Label','Set Fixed Axis', 'Tag', 'mToolsZoomFixed', ...
    'Callback', 'kazan(''mTools'',gcbo,[],guidata(gcbo))', ...
    'Checked', app.FixedZoomOpt);
  app.mCoord = uimenu(app.mTools, 'Label','Measure', 'Tag', 'mCoord', 'Separator', 'on', ...
    'Callback', 'kazan(''mTools'',gcbo,[],guidata(gcbo))', ...
    'Accelerator', 'E');
  % menu Plugins
  app.mPluginsMain = uimenu(app.MainFigure, 'Label','Plugins', 'Tag', 'mPluginsMain', 'Separator', 'off');
  % menu Window
  app.mWindowMain = uimenu(app.MainFigure, 'Label','Window', 'Tag', 'mWindowMain');
  app.mFileNewWindow = uimenu(app.mWindowMain, 'Label','New Window', 'Tag', 'mFileNewWindow', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))', 'Accelerator', 'K');
  app.mToolsShowinFigure = uimenu(app.mWindowMain, 'Label','Show in Separate Figure', 'Tag', 'mToolsShowinFigure', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.mPlot1DMenu = uimenu(app.mWindowMain, 'Label','Plot 1D', 'Tag', 'mPlot1DMenu', 'Separator', 'on');
  app.mOpenInFigure = uimenu(app.mPlot1DMenu, 'Label','Open figure', 'Tag', 'mOpenInFigure', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.mAxis1Of12 = uimenu(app.mPlot1DMenu, 'Label','1 / [1 2]', 'Tag', 'mAxis1Of12', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,112,guidata(gcbo))');
  app.mAxis2Of12 = uimenu(app.mPlot1DMenu, 'Label','2 / [1 2]', 'Tag', 'mAxis2Of12', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,212,guidata(gcbo))');
  app.mAxis1Of21 = uimenu(app.mPlot1DMenu, 'Label','1 / [2 1]', 'Tag', 'mAxis1Of21', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,121,guidata(gcbo))');
  app.mAxis2Of21 = uimenu(app.mPlot1DMenu, 'Label','2 / [2 1]', 'Tag', 'mAxis2Of21', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,221,guidata(gcbo))');
  app.Plot2DMenu = uimenu(app.mWindowMain, 'Label','Plot 2D', 'Tag', 'Plot2DMenu');
  app.m2DSurf = uimenu(app.Plot2DMenu, 'Label','Surf', 'Tag', 'm2DSurf', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.m2DScatter = uimenu(app.Plot2DMenu, 'Label','Scatter3', 'Tag', 'm2DScatter', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.m2DSurface = uimenu(app.Plot2DMenu, 'Label','Surface', 'Tag', 'm2DSurface', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.mToolsSkyline = uimenu(app.Plot2DMenu, 'Label','Show 2D density skyline plot', 'Tag', 'mToolsSkyline', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.mToolsSkyline1 = uimenu(app.Plot2DMenu, 'Label','Show 2D contour skyline plot', 'Tag', 'mToolsSkyline1', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.mToolsAssociatedwindow = uimenu(app.mWindowMain, 'Label','No Figure Is Associated', 'Tag', 'mToolsAssociatedwindow', 'Separator', 'on');
  app.mToolsAssociatedWindowReset = uimenu(app.mWindowMain, 'Label','Reset', 'Tag', 'mToolsAssociatedWindowReset', ...
    'Callback', 'kazan(''WindowsMenuSelected'',gcbo,[],guidata(gcbo))');
  app.mToolsAssociatedWindowSet = uimenu(app.mWindowMain, 'Label','To Set use KV_Tools/Associate of any KV-generated figure', 'Tag', 'mToolsAssociatedWindowSet', ...
    'Callback', '');
  
  % menu Help
  app.mHelp = uimenu(app.MainFigure, 'Label','Help', 'Tag', 'mHelp', 'Separator', 'off');
  app.mHelp3 = uimenu(app.mHelp, 'Label','How to Start', 'Tag', 'mHelp', 'Separator', 'off', 'Callback', ...
    'kazan(''mHelp'',gcbo,[],guidata(gcbo))');
  app.mHelp4 = uimenu(app.mHelp, 'Label','Data Load Routines', 'Tag', 'mHelp', 'Separator', 'off', 'Callback', ...
    'kazan(''mHelp'',gcbo,[],guidata(gcbo))');
  app.mHelp1 = uimenu(app.mHelp, 'Label','Axis Data Structure (ax)', 'Tag', 'mHelp', 'Separator', 'off', 'Callback', ...
    'kazan(''mHelp'',gcbo,[],guidata(gcbo))');
  app.mHelp2 = uimenu(app.mHelp, 'Label','Mouse Cursor', 'Tag', 'mHelp', 'Separator', 'off', 'Callback', ...
    'kazan(''mHelp'',gcbo,[],guidata(gcbo))');
  app.mHelp5 = uimenu(app.mHelp, 'Label','Plugins', 'Tag', 'mHelp', 'Separator', 'off', 'Callback', ...
    'kazan(''mHelp'',gcbo,[],guidata(gcbo))');
  app.mHelp6 = uimenu(app.mHelp, 'Label','How to Print', 'Tag', 'mHelp', 'Separator', 'off', 'Callback', ...
    'kazan(''mHelp'',gcbo,[],guidata(gcbo))');
  app.mHelpAbout = uimenu(app.mHelp, 'Label','About Kazan viewer', 'Tag', 'mAbout', 'Separator', 'on', 'Callback', ...
    'kazan(''mAbout'',gcbo,[],guidata(gcbo))');
  %**********************************************************************
  %**********************************************************************
  %**********************************************************************
  
  %**********************************************************************
  %*****************   C o n t e x t     m e n u    *********************
  %**********************************************************************
  app.cmContextMenu1 = uicontextmenu('Parent', app.MainFigure);
  app.cmNormal = uimenu(app.cmContextMenu1, 'Label','Normal', 'Tag', 'cmNormal', ...
    'Checked', 'on', ...
    'Callback','kazan(''cmContexMenu1'',gcbo,[],guidata(gcbo))');
  app.cmDiff1 = uimenu(app.cmContextMenu1, 'Label','Ps.mod., (ps_mod)', 'Tag', 'cmDiff1', ...
    'Callback', 'kazan(''cmContexMenu1'',gcbo,[],guidata(gcbo))');
  app.cmIntegrate = uimenu(app.cmContextMenu1, 'Label','Integrate', 'Tag', 'cmIntegrate', ...
    'Callback', 'kazan(''cmContexMenu1'',gcbo,[],guidata(gcbo))');
  app.cmNofilter = uimenu(app.cmContextMenu1, 'Label','Not Filtered', 'Tag', 'cmNofilter', ...
    'Separator', 'on', 'Checked', 'on', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmRCfilter = uimenu(app.cmContextMenu1, 'Label','RC (tc)', 'Tag', 'cmRCfilter', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmMvgaver = uimenu(app.cmContextMenu1, 'Label','M.Aver (tc)', 'Tag', 'cmMvgaver', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmSavGol = uimenu(app.cmContextMenu1, 'Label','Sav-Gol (tc, pol)', 'Tag', 'cmSavGol', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmApplyTo = uimenu(app.cmContextMenu1, 'Label','Apply to:', 'Tag', 'cmApplyTo');
  %**********************************************************************
  app.cmFiltAuto = uimenu(app.cmApplyTo, 'Label','Auto', 'Tag', 'cmFiltAuto', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmFilt1D = uimenu(app.cmApplyTo, 'Label','First Dim', 'Tag', 'cmFilt1D', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmFilt2D = uimenu(app.cmApplyTo, 'Label','Second Dim', 'Tag', 'cmFilt2D', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  app.cmFilt1D2D = uimenu(app.cmApplyTo, 'Label','Both', 'Tag', 'cmFilt1D2D', ...
    'Callback',  'kazan(''Filter_CallBack'',gcbo,[],guidata(gcbo))');
  %**********************************************************************
  app.cmxy = uimenu(app.cmContextMenu1, 'Label','x-y', 'Tag', 'cmxy', ...
    'Separator', 'on', 'Callback', ...
    'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.xxxxxx = uimenu(app.cmContextMenu1, 'Label','X', 'Tag', 'xxxxxx');
  %**********************************************************************
  app.cmxLinear = uimenu(app.xxxxxx, 'Label', 'X', 'Tag', 'cmxLinear', ...
    'Checked', 'on', 'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmxPoints = uimenu(app.xxxxxx, 'Label', 'X Points', 'Tag', 'cmxPoints', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmReverseX = uimenu(app.xxxxxx, 'Label', 'Reverse X', 'Tag', 'cmReverseX', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmSemilogX = uimenu(app.xxxxxx, 'Label', 'X Log', 'Tag', 'cmSemilogX', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmGspace = uimenu(app.xxxxxx, 'Label','X g-space (freq1)', 'Tag', 'cmGspace', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmxPPM = uimenu(app.xxxxxx, 'Label','X PPM (reference)', 'Tag', 'cmxPPM', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmxPPMmin1 = uimenu(app.xxxxxx, 'Label','X PPM-1 (reference)', 'Tag', 'cmxPPMmin1', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  %**********************************************************************
  app.yyyyyy = uimenu(app.cmContextMenu1, 'Label','Y', 'Tag', 'yyyyyy');
  %**********************************************************************
  app.cmyLinear = uimenu(app.yyyyyy, 'Label', 'Y', 'Tag', 'cmyLinear', ...
    'Checked', 'on', 'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmyPoints = uimenu(app.yyyyyy, 'Label', 'Y Points', 'Tag', 'cmyPoints', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmReverseY = uimenu(app.yyyyyy, 'Label', 'Reverse Y', 'Tag', 'cmReverseY', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmSemilogY = uimenu(app.yyyyyy, 'Label', 'Y Log', 'Tag', 'cmSemilogY', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmyPPM = uimenu(app.yyyyyy, 'Label','Y PPM (reference)', 'Tag', 'cmyPPM', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  app.cmyPPMmin1 = uimenu(app.yyyyyy, 'Label','Y PPM-1 (reference)', 'Tag', 'cmyPPMmin1', ...
    'Callback', 'kazan(''cmContexMenu2'',gcbo,[],guidata(gcbo))');
  %**********************************************************************
  %**********************************************************************
  %**********************************************************************
  
  % left panel
  app.fDirBrowser = uicontrol(app.MainFigure,'Style', 'frame', 'Tag', 'fDirBrowser', ...
    'Units','normalized');
  app.ePath = uicontrol(app.MainFigure,'Style', 'edit', 'Tag', 'ePath', ...
    'Units','normalized', 'Callback', 'kazan(''eDirectory_Callback'',gcbo,[],guidata(gcbo))');
  app.ddDataType = uicontrol(app.MainFigure,'Style', 'popupmenu', 'Tag', 'ddDataType', ...
    'Units','normalized', 'String', '*', 'Callback', 'kazan(''ddDataTypeValueChanged'',gcbo,[],guidata(gcbo))');
  app.pbUp = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'pbUp', ...
    'Units','normalized', 'String', 'Up', 'Callback', 'kazan(''pbUp_Callback'',gcbo,[],guidata(gcbo))');
  app.lDirlist = uicontrol(app.MainFigure,'Style', 'listbox', 'Tag', 'lDirlist', ...
    'Units','normalized', 'String', 'ss', 'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
  app.ddExt = uicontrol(app.MainFigure,'Style', 'popupmenu', 'Tag', 'ddExt', ...
    'Units','normalized', 'String', '*', 'Callback', 'kazan(''ddExtValueChanged'',gcbo,[],guidata(gcbo))');
   
  % Upper panel
  app.fGrOptions = uicontrol(app.MainFigure,'Style', 'frame', 'Tag', 'fGrOptions', 'Units','normalized');
  app.fGrOptions2 = uicontrol(app.MainFigure,'Style', 'frame', 'Tag', 'fGrOptions2', 'Units','normalized');
  app.rbSource = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'rbSource', 'Value', 1, 'Units','normalized', ...
    'String', 'Source', 'Callback', 'kazan(''SetDataSource'',1,guidata(gcbo))');
  app.rbOutput = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'rbOutput', 'Value', 0, 'Units','normalized',...
    'String', 'Output', 'Callback', 'kazan(''SetDataSource'',2,guidata(gcbo))');
  app.rbOutToSrc = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'rbOutToSrc', 'Units','normalized',...
    'String', ' << ', 'Callback', 'kazan(''mFile2Source'',gcbo,[],guidata(gcbo))');
  
  app.bReal = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'rbReal', 'Units','normalized', ...
    'String', 'Real', 'Callback', 'kazan(''SetDataPart'',1,guidata(gcbo))', 'Value', 1);
  app.bImag = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'rbImag', 'Units','normalized', ...
    'String', 'Re/Im', 'Callback', 'kazan(''SetDataPart'',2,guidata(gcbo))', 'Value', 0);
  app.rbMagnitude = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'rbMagnitude', 'Units','normalized', ...
    'String', 'Magn', 'Callback', 'kazan(''SetDataPart'',3,guidata(gcbo))', 'Value', 0);
  
  % plot type buttons
  app.tb1D = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'tb1D', 'Units','normalized', ...
    'String', '1D', 'Callback', 'kazan(''SetPlotType'',guidata(gcbo),1)', 'Value', 1);
  app.tbStacked = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'tbStacked', 'Units','normalized', ...
    'String', 'Stack', 'Callback', 'kazan(''SetPlotType'',guidata(gcbo),2)', 'Value', 0);
  app.tbContour = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'tbContour', 'Units','normalized', ...
    'String', 'Contour', 'Callback', 'kazan(''SetPlotType'',guidata(gcbo),3)', 'Value', 0);
  app.tbDensity = uicontrol(app.MainFigure,'Style', 'toggle', 'Tag', 'tbDensity', 'Units','normalized', ...
    'String', 'Density', 'Callback', 'kazan(''SetPlotType'',guidata(gcbo),4)', 'Value', 0);
  
  % zoom buttons
  app.tbMeasure = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'tbMeasure', 'Units','normalized', ...
    'String', '.', 'Tooltip', 'Measure', 'Callback', 'kazan(''mTools'',gcbo,[],guidata(gcbo))', 'Value', 0);
  app.tbX_half = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'tbX_half', 'Units','normalized', ...
    'String', '><', 'Tooltip', 'Zoom X', 'Callback', 'kazan(''Zoom_Callback'',gcbo,[],guidata(gcbo))', 'Value', 0);
  app.tbX_twice = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'tbX_twice', 'Units','normalized', ...
    'String', '<>', 'Tooltip', 'Zoom X', 'Callback', 'kazan(''Zoom_Callback'',gcbo,[],guidata(gcbo))', 'Value', 0);
  app.tbY_half = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'tbX_half', 'Units','normalized', ...
    'String', 'X', 'Tooltip', 'Zoom Y', 'Callback', 'kazan(''Zoom_Callback'',gcbo,[],guidata(gcbo))', 'Value', 0);
  app.tbY_twice = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'tbX_twice', 'Units','normalized', ...
    'String', '^', 'Tooltip', 'Zoom Y', 'Callback', 'kazan(''Zoom_Callback'',gcbo,[],guidata(gcbo))', 'Value', 0);
  
  % Main part
  app.tUnits = uicontrol(app.MainFigure,'Style', 'text', 'Tag', 'tUnits', 'Units','normalized');
  app.haReal = axes('Parent', app.MainFigure, 'Tag', 'haReal', 'Units','normalized', 'UIContextMenu', app.cmContextMenu1);
  app.haImag = axes('Parent', app.MainFigure, 'Tag', 'haImag', 'Units','normalized', 'UIContextMenu', app.cmContextMenu1);
  app.ddSelParameter = uicontrol(app.MainFigure,'Style', 'popupmenu', 'Tag', 'pmSelPar', 'Units','normalized', ...
    'Callback', 'kazan(''ddSelParameterValueChanged'',gcbo,[],guidata(gcbo))');
  app.eParameter =  uicontrol(app.MainFigure,'Style', 'edit', 'Tag', 'ePar', 'Units','normalized', ...
    'Callback', 'kazan(''eParameterValueChanged'',gcbo,[],guidata(gcbo))');
  app.sl2Dplot  = uicontrol(app.MainFigure,'Style', 'slider', 'Tag', 'sl2Dplot', 'Visible', 'off', ...
    'Units','normalized', 'Callback', 'kazan(''sl2Dplot_Callback'',gcbo,[],guidata(gcbo))');
  app.pb2DSelect  = uicontrol(app.MainFigure,'Style', 'pushbutton', 'Tag', 'pb2DSelect', 'Visible', 'off', 'String', 'X', 'Visible', false, ...
    'Units','normalized', 'Callback', 'kazan(''pb2DSelect_Callback'',gcbo,[],guidata(gcbo))');
  
  app.fMoveDataBr = uicontrol(app.MainFigure,'Style', 'frame', 'Tag', 'fMoveDataBr', ...
    'Callback', 'kazan(''fMoveDataBr_Callback'',gcbo, guidata(gcbo))','Units','normalized', ...
    'ButtonDownFcn', 'kazan(''MainFigure_ButtonDownFcn'', gcbo, [], guidata(gcbo))', 'String', '', ...
    'Enable', 'off');
  
  %   KAZAN VIEWER logo
  app.src.ax.xlabel = 'KAZAN viewer logo,';
  app.src.ax.type = 'data';
  sz = 0.3;
  kx = [-.15 0 0 0 1 0 1]*1.3; ky = [1 1 0 .5 1 .5 0];
  ax = [0 .5 .75 .25 .75 1]; ay = [0 1 .5 .5 .5 0];
  zx = [0 1 0  0  0 1 0 1 1 1]; zy = [0 1 1 .92 1 1  0 0 .08 0];
  nx = [0 0 1 1 1+.15]*1.4; ny = [0 1 0  1  1];
  app.src.ax.x = [8,8,-2,-2, kx-.3,ax+1.2,zx+2.4,ax+3.6,nx+4.8, ...
    8,8,6.2,8,8,-2,-2,-0.3,-2,-2]'*sz;
  app.src.ax.y = 1;
  app.src.y = [1,2,2,1,ky,ay,zy,ay,ny, ...
    1,0,0,0,-1,-1,0,0,0,1]';
  app.src = Verify(app, app.src);
  app.out{1} = app.src;
  
  % Initialize app structure
  app.LastLoadingScript = [];
  app.DataSource = 1;
  app.DataPart = 1;
  app.PlotType = 1;
  app.xrepresent = 'plot';
  app.yrepresent = 'plot';
  app.axisset = 'tight';
  app.axisaspect = 'normal';
  app.axisaspect1 = 'normal';
  app.axisgrid = 'off';
  app.axisgridminor = 'off';
  
  %   Viewer options
  app.ReadOptHandle = [];
  app.ReadOptLast = '';
  app.SecondCopy = 1;
  app.ext = '';
  app.Datatype = '';
  app.fname = 'none';
  app.fhistory = {};
  app.dhistory = {};
  app.process = '';
  app.filter = '';
  app.isScroller = 0;
  app.Projection = 1;
  app.AntiProjection = 2;
  app.Selection  = 1;
  app.DataPanelWidth = 0.2;
  app.AssociatedFigure = [];
  app.changePart = 0;
  
  app.View.Par = {};
  app.View.Par{end+1} = struct('title', 'xlabel', 'prefix', '', 'value', '', 'type', 's', 'about', '');
  app.View.Par{end+1} = struct('title', 'ylabel', 'prefix', '', 'value', '', 'type', 's', 'about', '');
  app.View.Par{end+1} = struct('title', 'Color', 'prefix', '', 'value', 'b', 'type', 's', 'about', '');
  app.View.Par{end+1} = struct('title', 'LineStyle', 'prefix', '', 'value', '-', 'type', 's', 'about', '');
  app.View.Par{end+1} = struct('title', 'LineWidth', 'prefix', '', 'value', 1, 'type', 'f', 'about', '');
  app.View.Par{end+1} = struct('title', 'Marker', 'prefix', '', 'value', 'none', 'type', 's', 'about', '');
  app.View.Par{end+1} = struct('title', 'dx', 'prefix', '', 'value', 0, 'type', 'f', 'about', '');
  app.View.Par{end+1} = struct('title', 's', 'prefix', '', 'value', 0, 'type', 'f', 'about', '');
  app.View.Par{end+1} = struct('title', 'ps_mod', 'prefix', '', 'value', 1, 'type', 'f', 'about', '');
  app.View.Par{end+1} = struct('title', 'tc', 'prefix', '', 'value', 2, 'type', 'f', 'about', '');
  app.View.Par{end+1} = struct('title', 'contour', 'prefix', '', 'value', '.2:.1:.6', 'type', 'f', 'about', '');
  app.View.Par{end+1} = struct('title', 'yslicelabel', 'prefix', '', 'value', '', 'type', 's', 'about', '');
  str = cellfun(@(s)s.title, app.View.Par, 'UniformOutput', false);
  set(app.ddSelParameter, 'String', str, 'Value', 1)
  
  app.DataParameters = app.ParameterReader('options');
  
  % Data extensions
  set(app.ddExt, 'String', app.ScriptLoad('extension'));
  
  % Load last saved position
  app.ini.KazanViewer.last_path = safeget(app.ini.KazanViewer, 'last_path', pwd);
  last_extension = safeget(app.ini.KazanViewer, 'last_extension', '');
  if any(strcmp(app.ddExt.String, last_extension))
    ddValue(app.ddExt, last_extension);
  end
  app.ePath.String = app.ini.KazanViewer.last_path;
  app.dir_path = app.ePath.String;
  
  set(app.MainFigure, 'ResizeFcn', 'kazan(''MainFigure_ResizeFcn'', gcbo,[],guidata(gcbo))');
  set(app.MainFigure, 'Resize', 'on');
  
  %     set(app.ePath, 'TooltipString', 'dir');
  app = ddExtValueChanged(gcbo, [], app, varargin);
  
  set(app.MainFigure, 'CreateFcn', 'kazan(''MainFigure_CreateFcn'', gcbo,[],guidata(gcbo))');
  set(app.MainFigure, 'DeleteFcn', 'kazan(''MainFigure_DeleteFcn'', gcbo,[],guidata(gcbo))');
  set(app.MainFigure, 'WindowButtonDownFcn', 'kazan(''MainFigure_ButtonDownFcn'', gcbo,[],guidata(gcbo))');
  set(app.MainFigure, 'WindowButtonUpFcn', 'kazan(''MainFigure_ButtonUpFcn'', gcbo,[],guidata(gcbo))');
  set(app.MainFigure, 'WindowButtonMotionFcn', 'kazan(''MainFigure_ButtonMotionFcn'', gcbo,[],guidata(gcbo))');
  
  app.NeedGUIRebuilding = true;
  
  %   app.t = timer('StartDelay', 4, 'Period', 5, 'ExecutionMode', 'fixedDelay', 'TasksToExecute',50);
  %   app.t.TimerFcn = @(x,y) kazan('TimerFunction', gcbo);
  %   start(app.t);
  
  guidata(fig, app);
  
  if nargout > 0
    varargout{1} = fig;
  end
  
  if isempty(oldfig)
    MainColor =  [0.8314, 0.8157, 0.7843];
    set(app.MainFigure, 'Name', 'KAZAN viewer');
  else
    MainColor = [0.76, 0.734, 0.682];
    set(app.MainFigure, 'Name',sprintf('[%d] KAZAN viewer',length(oldfig)+1));
  end
  set(app.MainFigure, 'Color',MainColor);
  MainFigure_ResizeFcn(0, 0, app);
  hdl = get(app.MainFigure, 'Children');
  for k=1:length(hdl)
    if strcmp(get(hdl(k), 'Type'), 'uicontrol')
      set(hdl(k), 'BackgroundColor', MainColor);
    end
  end
  SetPlotType(app, 2);
  
  MakeParameters(app, 'update_list');
  MakePlugins(app, 'search', []);
  MakeFavorites(app, 'update');
  
elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK
  
  try
    if (nargout)
      [varargout{1:nargout}] = feval(varargin{:}); % FEVAL switchyard
    else
      feval(varargin{:}); % FEVAL switchyard
    end
  catch err, ShowError(err);
  end
end

% --------------------------------------------------------------------
function value = ddValue(drop_down, varargin)
if nargin == 1
  str = drop_down.String;
  idx = drop_down.Value;
  value = str{idx};
else
  str = drop_down.String;
  value = varargin{1};
  idx = find(strcmp(str, value));
  if ~isempty(idx), drop_down.Value = idx; end
end

% --------------------------------------------------------------------
function TimerFunction(app)
% lDirlist_Callback(app.lDirlist, [], app, [])
% --------------------------------------------------------------------

function ViewFont_Callback(hObject, app)

app.DefaultFont = hObject.UserData;
set([app.mViewFont6,app.mViewFont7,app.mViewFont8,app.mViewFont9,app.mViewFont10,app.mViewFont11,app.mViewFont12], 'Checked', 'off')
set(hObject, 'Checked', 'on')
guidata(hObject, app);
MainFigure_ResizeFcn(hObject,[],app);

% --------------------------------------------------------------------

function ShowError(err, message)
if nargin > 1
  fprintf('\nKAZAN-%s error:', message);
else
  fprintf('\nKAZAN-intercepted error: ');
end
fprintf(2,'%s\n', err.message);
for ii=length(err.stack):-1:1
  try ref = which(err.stack(ii).name); catch, ref='????'; end
  fprintf('Stack <a href="matlab:opentoline(''%s'',%i)">%s: line %i</a>\n', ref, err.stack(ii).line, err.stack(ii).name, err.stack(ii).line);
end

% --------------------------------------------------------------------
function varargout = MainFigure_CreateFcn(~, ~, ~, varargin)
varargout = {};

% --------------------------------------------------------------------
function mAbout(~, ~, ~)
msgbox({'KAZAN data viewer and plugins';'Version 2.5.0 (06.17.2023)'; ...
  'Distributed under MIT License'; ...
  'based on KAZAN by Boris Epel and Alexey Silakov, 2003-06'; ...
  'Boris Epel, 2003 - 2023'}, ...
  'About', 'help')

% --------------------------------------------------------------------
function fig_size = check_figure_size(fig)
% manual resize of the figure
fig_size = get(fig, 'Position');

% minimum size check
update = 0;
if fig_size(3) < 0.3, fig_size(3) = 0.3; update = 1; end
if fig_size(4) < 0.3, fig_size(4) = 0.3; update = 1; end
if fig_size(2) + fig_size(4) > 1.0
  fig_size(2) = 1.0 - fig_size(4); update = 1;
end

if update
  %   set(app.MainFigure, 'Position', fig_size);
end

% --------------------------------------------------------------------
function MainFigure_DeleteFcn(~, ~, app, varargin)
app.ini.KazanViewer.autoZoomOff = app.autoZoomOff;
app.ini.KazanViewer.FixedZoomOpt = app.FixedZoomOpt;

inifilename = fullfile(userpath, 'kazan.ini');
try inimanage(inifilename, app.ini); catch err, ShowError(err, 'Store configuration'); end
MakePlugins(app, 'closeall', []);

% --------------------------------------------------------------------

function MainFigure_ResizeFcn(~, ~, app, varargin)

% change font size
standard_size = [0.324, 0.408, 0.406, 0.45];
fig_size = get(app.MainFigure, 'Position');
scaling_factor = fig_size(4) / standard_size(4);
DefaultFont = safeget(app, 'DefaultFont', 8);
font_size = DefaultFont + (scaling_factor-1) * (DefaultFont-2);
kazan_size_change(app.MainFigure, font_size);
set(0, 'DefaultUIControlFontSize', font_size);

app.NeedGUIRebuilding = false;

% if app.NeedGUIRebuilding
GUIResizeFcn([], [], app, varargin);
%   guidata(app.MainFigure, app);
% end
% --------------------------------------------------------------------

function GUIResizeFcn(~, ~, app, varargin)

% protection against out of GUI calls
if ~isfield(app, 'haReal'), return; end

% DataPanelWidth                     GraphSize
% -----------------------------------------------------------
% |           ||               button_panel1                 |
% |           ||---------------------------------------------|
% |           ||               button_panel2                 |
% |           ||---------------------------------------------|
% |           ||                                             |
% |           ||                                             |
% |           ||                axis_panel                   |
% |           ||                                             |
% |           ||                                             |
% |           ||                                             |
% |           ||---------------------------------------------|
% |           ||               parameters_panel              |
% -----------------------------------------------------------

fig_size = get(app.MainFigure, 'Position');
scaling_factor_width = fig_size(3) / 0.3;
scaling_factor_height = fig_size(4) / 0.4;

Border = 0.002;
ButtonPanelHeight = 0.04 + (1/max(scaling_factor_height, 0.4))*0.025;
Slider_Width = 0.01;
DataPanelWidth = safeget(app,'DataPanelWidth',0.2);
GraphSize = 1 - DataPanelWidth - Slider_Width - Border;

data_panel     = [0,0,DataPanelWidth,1];
slider         = [data_panel(1)+data_panel(3), 0,Slider_Width,1];
graphics_panel = [slider(1)+slider(3),         0,GraphSize,   1];
button_panel1  = [graphics_panel(1), 1-ButtonPanelHeight, GraphSize, ButtonPanelHeight];
ButtonPanelHeight1 = ButtonPanelHeight * 0.75;
button_panel2  = [graphics_panel(1), 1-ButtonPanelHeight-ButtonPanelHeight1, GraphSize, ButtonPanelHeight1];
parameters_panel  = [graphics_panel(1), 0, GraphSize-0.1, ButtonPanelHeight*0.8];
axis_panel     = graphics_panel + [0,ButtonPanelHeight/2,0,-(ButtonPanelHeight*2.4+ButtonPanelHeight1)];

% Update of Graphics Buttons panel
Button_Width = GraphSize / 9.95;
set(app.fGrOptions, 'Position', button_panel1);
rect_button = button_panel1 + Border*[1,1,-2,-2];
rect_button(3) = Button_Width - Border;
set(app.rbSource, 'Position', rect_button); % button Source
rect_button(1) = rect_button(1) + Button_Width;
rect_button(3) = Button_Width*0.5 - Border;
set(app.rbOutToSrc, 'Position', rect_button); % button Output to Source
rect_button(1) = rect_button(1) + Button_Width*0.5;
rect_button(3) = Button_Width-Border;
set(app.rbOutput, 'Position', rect_button); % button Output
rect_button(1) = rect_button(1) + Button_Width*1.2;
set(app.bReal, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.bImag, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.rbMagnitude, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width*1.2;
set(app.tb1D, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbStacked, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbContour, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbDensity, 'Position', rect_button);

% Update of BUTTON PANEL 2 / Zoom Buttons
set(app.fGrOptions2, 'Position', button_panel2);
Button_Width = GraphSize / 21.4;
ButtonSpace = Border;
rect_button = button_panel2 + Border*[2,1.25,1,-3];
rect_button(1) = button_panel2(1)+button_panel2(3) - 5*Button_Width;
rect_button(3) = Button_Width-ButtonSpace;
set(app.tbMeasure, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbX_twice, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbX_half, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbY_twice, 'Position', rect_button);
rect_button(1) = rect_button(1) + Button_Width;
set(app.tbY_half, 'Position', rect_button);

% parameters position
rect_button = button_panel2 + Border*[1,1,0.5,-2];
rect_button(3) = 0.12 - Border;
set(app.ddSelParameter, 'Position', rect_button);
rect_button(1) = rect_button(1) + 0.12 + Border;
rect_button(3) = button_panel2(3) - 5.2*Button_Width - rect_button(3) - 3*Border;
set(app.eParameter, 'Position', rect_button);

% draw axis
isScroller = app.isScroller;
Scroller_Width = 0.025;
ScrollerPlace = (Scroller_Width) * (isScroller > 0) + Border;
Border = Border * 2;
TextBorderX = Border + 0.06;
TextBorderY = 0.06;

switch app.DataPart
  case {2}
    WindowHeight = axis_panel(4)/2;
    rectG = axis_panel + [TextBorderX, TextBorderY, -ScrollerPlace-TextBorderX-2*Border, -TextBorderY-WindowHeight];
    set(app.haImag, 'Visible', 'on');
    set(app.haImag, 'Position', rectG);
    rectG(2) = rectG(2) + WindowHeight;
    set(app.haReal, 'Position', rectG);
  case {1,3}
    set(app.haImag, 'Visible', 'off');
    rectG = axis_panel + [TextBorderX, TextBorderY, -ScrollerPlace-TextBorderX-2*Border, -TextBorderY];
    set(app.haReal, 'Position', rectG);
end

% unit position
rect_button = parameters_panel + Border*[1,0.5,0,-1];
rect_button(3) = parameters_panel(1)+parameters_panel(3) - rect_button(1) - ScrollerPlace;
set(app.tUnits, 'Position', rect_button);
app = DataLoadOptions(app);

rectSl([2,4]) = axis_panel([2,4]) + [0.03, -0.03] ;
rectSl(1) = axis_panel(1) + axis_panel(3)- Scroller_Width - Border;
rectSl(3) = Scroller_Width;
set(app.sl2Dplot, 'Position', rectSl);
rectSel = [rectSl(1), rectSl(2)-0.04, Scroller_Width, 0.03];
set(app.pb2DSelect, 'Position', rectSel);

% --------------------------------------------------------------------
function app = ddExtValueChanged(h, eventdata, app, varargin)
val = ddValue(app.ddExt);
types = app.ScriptLoad('subtype', val);
set(app.ddDataType, 'String', types, 'Value', 1);
app.ext = val;
app = ddDataTypeValueChanged(h, eventdata, app, varargin);
app = LoadDirectory(app, '');
guidata(app.MainFigure, app);

% --------------------------------------------------------------------
% this function prepares correct place for different load options
function app = DataLoadOptions(app)

Border = 0.005;
controlHeight = 0.035;
DataPanelWidth = app.DataPanelWidth;

% update of Data read panel
PanelRect = [Border, 0, DataPanelWidth-2*Border, 1];
set(app.fDirBrowser, 'Position', PanelRect);
set(app.fMoveDataBr, 'Position', [DataPanelWidth-Border, 0, 0.01, 1]);
rectDir = PanelRect + [Border, PanelRect(4)-Border-controlHeight, -2*Border, -PanelRect(4) + controlHeight];
set(app.ePath, 'Position', rectDir);
rectDir(2) = rectDir(2) - controlHeight - Border;
rectDir(3) = (DataPanelWidth-6*Border)/2;
set(app.ddDataType, 'Position', rectDir);
rectDir(1) = rectDir(1) + DataPanelWidth/2-Border;
rectDir(3) = rectDir(3)-controlHeight-Border;
set(app.ddExt, 'Position', rectDir);
rectDir(1) = DataPanelWidth-2*Border-controlHeight;
rectDir(3) = controlHeight;
rectDir(4) = rectDir(4);
set(app.pbUp, 'Position', rectDir);

set(app.fDirBrowser, 'Position', PanelRect);
PanelRect = PanelRect + [Border, Border, -2*Border, -2*controlHeight-4*Border];

% Long control
controlLeft  = PanelRect(1);
controlWidth = DataPanelWidth-4*Border;
controlTop = PanelRect(2) + (0:3)*(controlHeight + Border);

% Two controls
controlLeft2  = controlLeft + controlWidth/2 + Border;
controlWidth2 = controlWidth/2 - Border;

% One description and two controls
controlLeft32  = controlLeft + controlWidth/3 + Border;
controlLeft33  = controlLeft + 2*controlWidth/3 + Border;
controlWidth3  = (controlWidth-2*Border)/3 - Border;

callback = 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))';

otype = [app.ext app.Datatype];
if ~strcmp(app.ReadOptLast, otype)
  delete(app.ReadOptHandle)
  app.ReadOptHandle = [];
end
app.ReadOptLast = otype;
switch app.ReadOptLast
  case {'*.*ASCII', '*.datASCII', '*ASCII', '*.txtASCII', '*.ASCII'}
    spaceneeded = 2*controlHeight+2*Border;
    
    % create three controls for unit, delimiter and read format
    if isempty(app.ReadOptHandle) % create
      app.ReadOptHandle(1) = uicontrol(app.MainFigure, 'Style', 'popupmenu',...
        'String', {'tab', 'space', ',', ';'}, ...
        'UserData', {'\t', ' ', ',', ';'}, 'Units', 'normalized', ...
        'Position', [controlLeft,controlTop(1),controlWidth2,controlHeight], ...
        'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
      app.ReadOptHandle(2) = uicontrol(app.MainFigure, 'Style', 'popupmenu',...
        'String', {'Y', 'rY, iY', 'X,Y', 'X,rY,iY', 'idx, X, Y', 'idx, X, rY, iY', 'X,Y1,Y2,Y3..'}, 'Units', 'normalized', ...
        'UserData', {1,2,3,4,5,6,7}, ...
        'Value', 4,'Position', [controlLeft2,controlTop(1),controlWidth2,controlHeight], ...
        'Callback', callback);
      app.ReadOptHandle(3) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', 'Time, s', 'Units', 'normalized', ...
        'Position', [controlLeft,controlTop(2),controlWidth,controlHeight], ...
        'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
      guidata(app.MainFigure, app);
    else % resize
      set(app.ReadOptHandle(1), 'Position', [controlLeft,controlTop(1),controlWidth2,controlHeight]);
      set(app.ReadOptHandle(2), 'Position', [controlLeft2,controlTop(1),controlWidth2,controlHeight]);
      set(app.ReadOptHandle(3), 'Position', [controlLeft,controlTop(2),controlWidth,controlHeight]);
    end
  case {'*.RMN_2D','*.rmnRMN_2D'}
    spaceneeded = 1*controlHeight+1*Border;
    app.ReadOptHandle(1) = uicontrol(app.MainFigure, 'Style', 'popupmenu',...
      'String', {'D1:Time D2:Time', 'D1:Time D2:Freq', 'D1:Freq D2:Time', 'D1:Freq D2:Freq'}, ...
      'Units', 'normalized', 'UserData', {'2DTT', '2DTF', '2DFT', '2DFF'}, ...
      'Position', [controlLeft,controlTop(1),controlWidth,controlHeight], ...
      'Callback', callback);
    guidata(app.MainFigure, app);
  case '*OPUS_FTIR'
    spaceneeded = 1*controlHeight+1*Border;
    app.ReadOptHandle(1) = uicontrol(app.MainFigure, 'Style', 'popupmenu',...
      'String', {'Interferogram', 'Baseline (fft)', 'Data-Baseline(fft)'}, ...
      'Units', 'normalized', 'UserData', [1, 2, 3], ...
      'Position', [controlLeft,controlTop(1),controlWidth,controlHeight], ...
      'Callback', callback);
    guidata(app.MainFigure, app);
  case {'*.datASCII2D', '*ASCII2D', '*.txtASCII2D', '*.ASCII2D'}
    spaceneeded = 4*controlHeight+4*Border;
    
    % create three controls for unit, delimiter and read format
    if isempty(app.ReadOptHandle) % create
      % control for unit
      app.ReadOptHandle(1) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', 'Time, s', 'Units', 'normalized', ...
        'Position', [controlLeft,controlTop(4),sz,controlHeight], ...
        'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
      % control for X: label, xsize, dwelltime
      app.ReadOptHandle(2) = uicontrol(app.MainFigure, 'Style', 'text',...
        'String', 'X: n,dw', 'Units', 'normalized', ...
        'Position', [controlLeft,controlTop(3),controlWidth3,controlHeight],...
        'Callback', callback);
      app.ReadOptHandle(3) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', '1', 'Units', 'normalized', ...
        'Position', [controlLeft32,controlTop(3),controlWidth3,controlHeight],...
        'Callback', callback);
      app.ReadOptHandle(4) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', '1', 'Units', 'normalized', ...
        'Position', [controlLeft33,controlTop(3),controlWidth3,controlHeight],...
        'Callback', callback);
      % control for Y: label, xsize, dwelltime
      app.ReadOptHandle(5) = uicontrol(app.MainFigure, 'Style', 'text',...
        'String', 'Y: n,dw', 'Units', 'normalized', ...
        'Position', [controlLeft,controlTop(2),controlWidth3,controlHeight],...
        'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
      app.ReadOptHandle(6) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', '1', 'Units', 'normalized', ...
        'Position', [controlLeft32,controlTop(2),controlWidth3,controlHeight],...
        'Callback', callback);
      app.ReadOptHandle(7) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', '1', 'Units', 'normalized', ...
        'Position', [controlLeft33,controlTop(2),controlWidth3,controlHeight],...
        'Callback', callback);
      % control for central frequency
      app.ReadOptHandle(8) = uicontrol(app.MainFigure, 'Style', 'text',...
        'String', 'C. Freq', 'Units', 'normalized', ...
        'Position', [controlLeft,controlTop(1),controlWidth2,controlHeight],...
        'Callback', callback);
      app.ReadOptHandle(9) = uicontrol(app.MainFigure, 'Style', 'edit',...
        'String', '0', 'Units', 'normalized', ...
        'Position', [controlLeft2,controlTop(1),controlWidth2,controlHeight],...
        'Callback', callback);
      guidata(app.MainFigure, app);
    else
      set(app.ReadOptHandle(1), 'Position', [controlLeft,controlTop(4),sz,controlHeight]);
      set(app.ReadOptHandle(2), 'Position', [controlLeft,controlTop(3),controlWidth3,controlHeight]);
      set(app.ReadOptHandle(3), 'Position', [controlLeft32,controlTop(3),controlWidth3,controlHeight]);
      set(app.ReadOptHandle(4), 'Position', [controlLeft33,controlTop(3),controlWidth3,controlHeight]);
      set(app.ReadOptHandle(5), 'Position', [controlLeft,controlTop(2),controlWidth3,controlHeight]);
      set(app.ReadOptHandle(6), 'Position', [controlLeft32,controlTop(2),controlWidth3,controlHeight]);
      set(app.ReadOptHandle(7), 'Position', [controlLeft33,controlTop(2),controlWidth3,controlHeight]);
      set(app.ReadOptHandle(8), 'Position', [controlLeft,controlTop(1),controlWidth2,controlHeight]);
      set(app.ReadOptHandle(9), 'Position', [controlLeft2,controlTop(1),controlWidth2,controlHeight]);
    end
  case {'*UofC_bin', '*.imgUofC_img'}
    spaceneeded = 1*controlHeight+1*Border;
    app.ReadOptHandle(1) = uicontrol(app.MainFigure, 'Style', 'popupmenu',...
      'String', {'Modula', 'LabView'}, ...
      'Units', 'normalized', 'UserData', [1, 2, 3], ...
      'Position', [controlLeft,controlTop(1),controlWidth,controlHeight], ...
      'Callback', 'kazan(''lDirlist_Callback'',gcbo,[],guidata(gcbo))');
    guidata(app.MainFigure, app);
  otherwise
    spaceneeded = 0;
end
% decrease space for directory list
PanelRect(2) = PanelRect(2) + spaceneeded;
PanelRect(4) = PanelRect(4) - spaceneeded;
set(app.lDirlist, 'Position', PanelRect);

MainColor =  get(app.MainFigure, 'Color');
set(app.ReadOptHandle, 'BackgroundColor', MainColor);

% ---------------------------------------------------
function MainFigure_ButtonDownFcn(~, ~, app)
switch get(app.MainFigure, 'SelectionType')
  case 'open'
    %     if strcmp(get(app.mToolsZoom, 'Checked'), 'off'), return; end
    app.Type = 0;
    app.MMove = 0;
    %     hdl = [app.mViewAxistight, app.mViewAxisauto app.mViewAxisfixed];
    %     hand = findobj(hdl, 'Checked', 'on');
    mViewAxisSet(app.mViewAxistight, [], app);
  case 'extend'
    if strcmp(get(app.mToolsZoom, 'Checked'), 'on'), return; end
    app.Type  = 1;
    app.MMove = 1;
    set(app.MainFigure, 'Pointer', 'crosshair');
  case 'normal' % for zoom
    if get(app.MainFigure, 'CurrentObject') == app.fMoveDataBr % Change width of Directory list
      app.changePart = 1; % Start change width of Directory list
      set(app.fMoveDataBr, 'BackgroundColor', [1, 0.9, 0.9]);
    else % zoom
      if strcmp(get(app.mToolsZoom, 'Checked'), 'off'), return; end
      app.Type  = 0;
      app.MMove = 0;
      [x, y, axhand] = getcoord(app);
      curval = get(app.MainFigure, 'CurrentPoint');
      if ~isempty(x)
        app.StartP = [x, y, curval];
        rbbox;
        [x, y] = getcoord(app, axhand);
        app.FinishP = [x, y];
        if app.FinishP(1) == app.StartP(1) || app.FinishP(2) == app.StartP(2), return; end
        if ~isempty(x)
          set(axhand, 'XLim', sort([app.StartP(1), app.FinishP(1)]), ...
            'YLim', sort([app.StartP(2), app.FinishP(2)]));
          %             if isfield(app, 'Rectang')
          %                 delete(app.Rectang);
          %
          %             end
          % switching zoom off boep maybe good maybe - not
          % I hope, this is golden middle. alsi
          if strcmp(app.autoZoomOff, 'on')
            mTools(app.mToolsZoom,[],app);
          end
          if strcmp(app.FixedZoomOpt, 'on')
            mViewAxisSet(app.mViewAxisfixed,[],app);
          end
        end
      end
    end
end
guidata(app.MainFigure, app);
% ---------------------------------------------------

function MainFigure_ButtonUpFcn(h, ~, app)

if app.changePart
  MainFigure_ResizeFcn(h, [], app);
  set(app.fMoveDataBr, 'BackgroundColor', [0.7, 0.7, 0.7]);
end
app.changePart = 0;
app.MMove = 0;

switch app.Type
  case 1
    set(app.MainFigure, 'Pointer', 'arrow');
end
guidata(app.MainFigure, app);

% ---------------------------------------------------
function [x,dx,unit] = get_abscissa(app)
switch app.DataSource
  case 1
    switch app.Projection
      case 1
      unit = safeget(app.src.ax, 'xunit', '');
      x = app.src.ax.x;
      dx = safeget(app.src.ax, 'dx', 0.0);
      otherwise
        unit = safeget(app.src.ax, 'yunit', '');
        x = app.src.ax.y;
        dx = safeget(app.src.ax, 'dy', 0.0);
    end
  otherwise
    switch app.Projection
      case 1
        unit = safeget(app.out{1}.ax, 'xunit', '');
        x = app.out{1}.ax.x;
        dx = safeget(app.out{1}.ax, 'dx', 0.0);
      otherwise
        unit = safeget(app.out{1}.ax, 'yunit', '');
        x = app.out{1}.ax.y;
        dx = safeget(app.out{1}.ax, 'dy', 0.0);
    end
end

% ---------------------------------------------------
function MainFigure_ButtonMotionFcn(~, ~, app)
% protection
if ~isfield(app, 'changePart'), return
end
app.MMove = safeget(app, 'MMove' , 0);
app.Type = safeget(app, 'Type' , 0);

try
if app.MMove
  switch app.Type
    case 1
      [x, y] = getcoord(app);
      if ~isempty(x)
        s = sprintf(' x= %g, y= %f',x,y);
        Set(app, 'cursor',s);
      end
  end
end
if app.changePart                          % do width of Directory list
  asd = get(app.MainFigure, 'CurrentPoint');
  mn_pos = get(app.MainFigure, 'Position');
  MBarPos = get(app.fMoveDataBr, 'position');
  if ~(asd(2)>mn_pos(4) || asd(2)<0 || asd(1)>mn_pos(3)-0.15 || asd(1) < 0.2)
    set(app.fMoveDataBr, 'BackgroundColor', [1, 0.9, 0.9]);
    app.DataPanelWidth = asd(1)-MBarPos(3)*0.5; % to make pointer on the middle of the bar
    guidata(app.MainFigure, app);
    app = DataLoadOptions(app);
  else
    set(app.fMoveDataBr, 'BackgroundColor', [0.1, 0.1, 0.1]);
  end
end
catch
end
guidata(app.MainFigure, app);

if isempty(app.Message) && toc(app.MessageBlocker) > 5
  try
  cursorPoint = get(app.haReal, 'CurrentPoint');
  [~,dx,unit] = get_abscissa(app);
  xinfo = '';
  if strcmp(unit,'G')
    gref = safeget(app.src.ax, 'freq1', 0)/1.399624504120092e6;
    if gref ~= 0
      xinfo = sprintf(' g=%6.5f', gref(1) / (cursorPoint(1,1) + dx));
    end
  elseif strcmp(unit, 'T')
    gref = safeget(app.src.ax, 'freq1', 0)/1.399624504120092e10;
    if gref ~= 0
      xinfo = sprintf(' g=%6.5f', gref(1) / (cursorPoint(1,1) + dx));
    end
  elseif strcmp(unit,'mT')
    gref = safeget(app.src.ax, 'freq1', 0)/1.399624504120092e7;
    if gref ~= 0
      xinfo = sprintf(' g=%6.5f', gref(1) / (cursorPoint(1,1) + dx));
    end
  end
  if app.sl2Dplot.Visible
    Set(app, 'cursor', sprintf('x:%7.5g %s y:%7.5g [%i]', cursorPoint(1,1), xinfo, cursorPoint(1,2), floor(app.sl2Dplot.Value+0.5)));
  else
    Set(app, 'cursor', sprintf('x:%7.5g %s y:%7.5g', cursorPoint(1,1), xinfo, cursorPoint(1,2)));
  end
  catch
  end
end

% --------------------------------------------------------------------
function OnFavoriteSelected(h, ~, app, varargin)
switch h.Label
  case 'Add Current', MakeFavorites(app, 'addcurrent');
  case 'Remove Current', MakeFavorites(app, 'removecurrent');
  case 'Remove All', MakeFavorites(app, 'removeall');
  otherwise
    if exist(h.Label, 'file')
      app.ePath.String = h.Label;
      eDirectory_Callback(app.ePath, [], app);
    end
end

% --------------------------------------------------------------------
function mFileSave(h, ~, app, varargin)
switch h
  case app.mFileOpen
    if exist('uigetdir', 'file')
      directory_name = uigetdir(pwd,'Select directory');
    else
      [~, directory_name] = uiputfile([pwd,'\select.dir'],'Select directory');
    end
    if ~directory_name, return; end
    LoadDirectory(app, directory_name);
  case {app.mFileSave, app.mFileSaveProc}
    curdir = pwd;
    cd(app.dir_path);
    [fname,pname,sel] = uiputfile({'*.txt', 'ASCII files (*.txt)'; '*.dat', 'ASCII files (*.dat)'; '*.d01', 'SpecMan files (*.d01)';'*.dsc', 'Bruker files (new, *.dsc)'; '*.par', 'Bruker files (old, *.par)'; '*.lmb', 'WinSIM files (*.lmb)'}, 'Select file format'); %(don''t forget to type extension !!!)
    cd(curdir);
    
    if fname == 0, return; end
    [~,~,ext] = fileparts(fname);
    src = app.src; src.fname = app.fname;
    
    if h == app.mFileSaveProc
      [src.y, src.ax] = processing(src.y, src.ax);
      src.ax = getplotaxis(app.xrepresent, app.yrepresent, src.ax);
      switch app.DataPart
        case 1, src.y = real(src.y);
        case 2, src.y = real(src.y);
        case 4, src.y = abs(src.y);
      end
    end
    
    switch sel
      case 3
        kv_d01write([pname, fname], src);
        disp(['file ',[pname, fname] ,' have been successfully saved in format "d01/exp"'])
      case 4
        kv_brukerwrite([pname, fname], src, 'DSC');
        disp(['file ',[pname, fname] ,' have been successfully saved in format "dsc/dta"'])
      case 5
        kv_brukerwrite([pname, fname], src, 'PAR');
        disp(['file ',[pname, fname] ,' have been successfully saved in format "par/spc"'])
      case 6
        kv_LMBwrite([pname, fname], src);
        disp(['file ',[pname, fname] ,' have been successfully saved in format "LMB"'])
      otherwise %1 or %2
        if isempty(ext), fname = [fname,ext]; end
        kv_asciiwrite([pname, fname], src, 'usefooter', 1, 'origin', fname, 'parameters', app.DataParameters);
        disp(['file ',[pname, fname] ,' successfully saved in format "',ext,'"'])
    end
  case app.mFileConvert
    button = questdlg('Converting all files to ASCII?', 'ASCII file conversion',...
      'Yes (dat)', 'Yes (txt)', 'No', 'No');
    if ~strcmp(button, 'No')
      for kk=1:length(app.file_names)
        if app.is_dir(kk), continue; end
        fname = fullfile(app.dir_path,app.file_names{kk});
        hdl = LoadFile(fullfile(app.dir_path,app.file_names{kk}), app);
        y = real(hdl.src.y);
        y1 = imag(hdl.src.y);
        x = hdl.src.ax.x + safeget(hdl.src.ax, 'dx', 0);
        [fpath,name] = fileparts(fname);
        if strcmp(button, 'Yes (dat)')
          fname = fullfile(fpath, [name,'.dat']);
        else
          fname = fullfile(fpath, [name,'.txt']);
        end
        if sum(y1)
          p = [x,y,y1];
        else
          p = [x,y];
        end
        save(fname, 'p', '-ascii')
        disp([fname,' successfully saved in format "dat"'])
      end
    end
end

% --------------------------------------------------------------------
function OnPluginSelected(h, ~, app, varargin)
switch h.Label
  case 'Arrange plugins'
    posMain = get(app.MainFigure, 'Position');
    kkx = 4; kky = 9;
    
    try
      for i=1:size(app.Plugins, 2)
        hPlugin = app.Plugins{i};
        posPlugin = get(hPlugin, 'Position');
        switch get(hPlugin, 'Units')
          case 'points'
            posPlugin(2) = (posMain(2) - 2*(i-1) - posPlugin(4)/kky + posMain(4))*kky;
            posPlugin(1) = (posMain(1) + posMain(3) + 1.2 + 8*(i-1))*kkx;
            set(hPlugin, 'Position', posPlugin);
          case 'characters'
            posPlugin(2) = posMain(2) - 2*(i-1) - posPlugin(4) + posMain(4);
            posPlugin(1) = posMain(1) + posMain(3) + 1.2 + 8*(i-1);
            set(hPlugin, 'Position', posPlugin);
        end
      end
    catch
      DeletePlugins(hPlugin, app.MainFigure);
      guidata(app.MainFigure, app);
    end
  case 'Reload plugins'
    MakePlugins(app, 'search', []);
  otherwise
    switch h.Parent.Label
      case 'Add to Favorites'
        plugin_name = upper(h.UserData);
        app.ini.(plugin_name).favorite = '1';
        inifilename = fullfile(userpath, 'kazan.ini');
        try inimanage(inifilename, app.ini); catch, end
        MakePlugins(app, 'search', []);
      case 'Remove from Favorites'
        plugin_name = upper(h.UserData);
        if isfield(app.ini, plugin_name)
          app.ini.(plugin_name).favorite = '0';
          inifilename = fullfile(userpath, 'kazan.ini');
          try inimanage(inifilename, app.ini); catch, end
          MakePlugins(app, 'search', []);
        end
      otherwise
        %         [~,guid] = fileparts(tempname);
        %         runstring = sprintf('%s(app, ''%s'')', h.UserData, guid);
        app.Plugins{end+1} = eval(h.UserData);
        gdata = guidata(app.Plugins{end});
        gdata.hh = app.MainFigure;
        guidata(app.Plugins{end}, gdata);
        
        [~,name] = fileparts(get(app.Plugins{end}, 'FileName'));
        str = [name, '(''FileUpdate'',app.Plugins{end},guidata(app.Plugins{end}));'];
        try eval(str); catch err, ShowError(err); end
        
        guidata(app.MainFigure, app);
        fprintf('Plugin is added. Overall %i\n', length(app.Plugins))
    end
end


% --------------------------------------------------------------------
function res = GetPlotNumber(app)
if app.DataSource ~= 1
  nplots = length(app.out);
else
  nplots = 0;
end
% if DataSource 1 or 3 print source
if app.DataSource~=2
  fplot = 0;
else
  fplot = 1;
end
res = fplot:nplots;

% --------------------------------------------------------------------
function PlotData(mainform, varargin)
try
  app = guidata(mainform);
  %   rotate3d(app.MainFigure,'off')
  
  % this part is used to redirect output to different plots
  if nargin < 2, haReal = app.haReal;
  else, haReal = varargin{1};
  end
  
  if nargin < 3, haImag = app.haImag;
  else, haImag = varargin{2};
  end
  
  if nargin < 4, fig = app.MainFigure;
  else, fig = varargin{3};
  end
  
  raxis = axis(haReal);
  iaxis = axis(haImag);
  newax.xlabel = '?,';
  newax.ylabel = '?,';
  
  %  clear all plots
  ch = allchild(haReal);
  delete(ch);
  ch = allchild(haImag);
  delete(ch);
  set(haReal, 'NextPlot', 'add');
  set(haImag, 'NextPlot', 'add');
  
  % plot image
  if strcmp(safeget(app.src.ax, 'type', 'data'), 'image') && (app.DataSource==1 || app.DataSource==3)
    image(app.src.y, 'Parent', haReal);
    set(haReal, 'YDir', 'reverse')
    axis(haReal, 'image');
  % plot 1D / 2D  
  else
    % protecting axis against upside down etc
    axis(haReal, 'normal');
    axis(haImag, 'normal');
    set(haReal, 'XDir', 'normal')
    set(haReal, 'YDir', 'normal')
    
    for ii = GetPlotNumber(app)
      % data selection
      if app.PlotType~=1
        proj = 3;
      else
        proj = app.Projection;
      end
      if ii~=0
        % output
        if isempty(app.out{ii}.y), continue; end
        tt  = GetSlice(app.out{ii}, proj, app.Selection);
        UserData = safeget(app.out{ii}.ax,'UserData',1);
      else
        % source
        if isempty(app.src.y), continue; end
        tt  = GetSlice(app.src, proj, app.Selection);
        UserData = 1;
      end
      if ~contains(class(tt.y), 'uint8')
        [y, ax] = processing(tt.y, tt.ax);
      else
        y = tt.y;
        ax = tt.ax;
      end
      newax = getplotaxis(app.xrepresent, app.yrepresent, ax);
      vertical_offset = safeget(ax, 's', 0.0);
      y = y + vertical_offset;

      % data printing
      switch app.PlotType
        case {1,2} % 1D and Stack
          switch app.DataPart
            case 1
              hhh = superplot(app.xrepresent, app.yrepresent, newax, real(y), haReal);
              set(hhh,'UserData',UserData)
            case 2
              hhh = superplot(app.xrepresent, app.yrepresent, newax, real(y), haReal);
              set(hhh,'UserData',UserData)
              hhh = superplot(app.xrepresent, app.yrepresent, newax, imag(y), haImag);
              set(hhh,'UserData',UserData)
              axis(haImag, app.axisset);
            case 3
              hhh = superplot(app.xrepresent, app.yrepresent, newax, abs(y), haReal);
              set(hhh,'UserData',UserData)
          end
          if app.PlotType==2 && isfield(newax,'yslicelabel') && ~isempty(newax.yslicelabel)
            xmin = min(newax.x) + safeget(newax, 'yslicelabeldx',0);
            xmax = max(newax.x) + safeget(newax, 'yslicelabeldx',0);
            ymax = y(end,:) + safeget(newax, 'yslicelabeldy',0);
            for jj=1:length(newax.y)
              text((xmax-xmin)*.9+xmin, ymax(jj), sprintf(newax.yslicelabel, newax.y(jj)), 'Parent', haReal);
            end
          end
        case 3 % contour
          set(0,'CurrentFigure', fig)
          if isfield(ax, 'contour'), cc = ax.contour;
          else, cc = .1:.1:.6;
          end
          
          switch app.DataPart
            case 1
              set(fig, 'CurrentAxes', haReal)
              maxyy=max(max(real(y))); minyy=min(min(real(y)));
              if maxyy < 0, sh = minyy; else, sh = 0; end
              contour(newax.x, newax.y, real(y.'), cc*maxyy+sh);
            case 2
              set(fig, 'CurrentAxes', haReal);
              maxyy=max(max(real(y))); minyy=min(min(real(y)));
              if maxyy < 0, sh = minyy; else, sh = 0; end
              contour(outx, newax.y, real(y.'), cc*(maxyy-minyy)+sh);
              set(fig, 'CurrentAxes', haImag);
              maxyy=max(max(imag(y))); minyy=min(min(imag(y)));
              if maxyy < 0, sh = minyy; else, sh = 0; end
              contour(newax.x, newax.y, imag(y.'), cc*(maxyy-minyy)+sh);
              axis(haImag, app.axisset);
            case 3
              set(fig, 'CurrentAxes', haReal)
              maxyy=max(max(abs(y)));
              contour(newax.x, newax.y, abs(y.'), cc*maxyy);
          end
        case 4 %density
          if isfield(ax, 'contour'), cc = ax.contour;
          else, cc = .1:.1:.6;
          end
          maxyy=max(max(y));
          minyy=min(min(y));
          if maxyy < 0, sh = minyy;
          else, sh = 0;
          end
          
          set(0,'CurrentFigure', fig)
          switch app.DataPart
            case 1
              set(fig, 'CurrentAxes', haReal)
              clim = cc*real(maxyy)+real(sh);
              imagesc(newax.x, newax.y, real(y.'), [min(clim), max(clim)]);
            case 2
              set(fig, 'CurrentAxes', haReal)
              clim1 = cc*real(maxyy-minyy)+real(sh);
              imagesc(newax.x, newax.y, real(y.'), [min(clim1), max(clim1)]);
              set(fig, 'CurrentAxes', haImag)
              clim2 = cc*imag(maxyy-minyy)+imag(sh);
              imagesc(newax.x,newax.y, imag(y.'), [min(clim2), max(clim2)]);
              axis(haImag, app.axisset);
            case 3
              set(fig, 'CurrentAxes', haReal)
              clim = cc*(max(max(abs(y)))-min(min(abs(y))));
              imagesc(newax.x, newax.y, abs(y.'), [min(clim), max(clim)]);
          end
        case 5 % 3d plot
          
          set(0,'CurrentFigure', fig)
          switch app.DataPart
            case 1
              set(fig, 'CurrentAxes', haReal)
              h = figure; hold on
              for jj=1:size(y,2)
                plot3(ax.x, ax.y(jj)*ones(length(ax.x),1), real(y(:,jj))); hold on
              end
              cameratoolbar(h);
              rotate3d(h,'on')
              view(45,25)              ;
            case 2
              set(fig, 'CurrentAxes', haReal)
              %               h = figure; hold on
              for jj=1:size(y,2)
                plot3(ax.x, ax.y(jj)*ones(length(ax.x),1), imag(y(:,jj))); hold on
              end
              cameratoolbar(h);
              rotate3d(h,'on')
              view(45,25)              ;
            case 3
              set(fig, 'CurrentAxes', haReal)
              h = figure; hold on
              for jj=1:size(y,2)
                plot3(ax.x, ax.y(jj)*ones(length(ax.x),1), abs(y(:,jj)))
              end
              cameratoolbar(h);
              rotate3d(h,'on')
              view(45,25)              ;
          end
        case 6 % for hyscorebox and image2data <<< AlSi 30.03.05 <<<
          if ii==0||strcmp(safeget(app.out{ii}, 'forceplot', 'contour'), 'image')  % input data
            if isfield(ax, 'contour'), cc = ax.contour;
            else, cc = .1:.1:.6;
            end
            maxyy=max(max(y));
            minyy=min(min(y));
            if maxyy < 0, sh = minyy;
            else, sh = 0;
            end
            
            set(0,'CurrentFigure', fig)
            if ~isempty(y)
              b = whos('y');
              if strcmp(b.class, 'uint8')
                image(newax.x, newax.y, y, 'Parent', haReal);
                %                         set(haReal, 'YDir', 'reverse')
                %                         axis(haReal, 'image');
              else
                switch app.DataPart
                  case 1
                    set(fig, 'CurrentAxes', haReal)
                    clim = cc*real(maxyy-minyy)+real(sh);
                    imagesc(newax.x, newax.y, real(y).', [min(clim), max(clim)]);
                    %               surf(ax.x, ax.y, real(y.'), 'LineStyle', 'none');
                  case 2
                    set(fig, 'CurrentAxes', haReal)
                    clim1 = cc*real(maxyy-minyy)+real(sh);
                    imagesc(newax.x, newax.y, real(y).', [min(clim1), max(clim1)]);
                    %                surf(ax.x, ax.y, real(y.'), 'LineStyle', 'none');
                    set(fig, 'CurrentAxes', haImag)
                    clim2 = cc*imag(maxyy-minyy)+imag(sh);
                    imagesc(newax.x, newax.y, imag(y).', [min(clim2), max(clim2)]);
                    %                surf(ax.x, ax.y, imag(y.'), 'LineStyle', 'none');
                    axis(haImag, app.axisset);
                  case 3
                    set(fig, 'CurrentAxes', haReal)
                    clim = cc*(max(max(abs(y)))-min(min(abs(y))));
                    imagesc(newax.x, newax.y, abs(y).', [min(clim), max(clim)]);
                    %                surf(ax.x, ax.y, abs(y.'), 'LineStyle', 'none');
                end
              end
            end
          else
            switch app.DataPart
              case 1
                superplot(app.xrepresent, app.yrepresent, ax, ax.y, haReal);
              case 2
                superplot(app.xrepresent, app.yrepresent, ax, ax.y, haReal);
                superplot(app.xrepresent, app.yrepresent, ax, ax.y, haImag);
                axis(haImag, app.axisset);
              case 3
                superplot(app.xrepresent, app.yrepresent, ax, ax.y, haReal);
            end
          end
      end
    end
  end
  if strcmp(get(app.mViewAxisfixed, 'Checked'), 'on')
    axis(haReal, raxis);
    axis(haImag, iaxis);
    %             axis(haReal, app.axisset);
    %             axis(haImag, app.axisset);
  else
    axis(haReal, app.axisset);
    axis(haImag, app.axisset);
  end
  axis(haReal, app.axisaspect);
  axis(haImag, app.axisaspect);
  if strcmp(app.axisaspect1, 'equal')
    axis(haReal, app.axisaspect1);
    axis(haImag, app.axisaspect1);
  end
  grid(haReal, app.axisgrid)
  grid(haImag, app.axisgrid)
  haReal.XMinorGrid = app.axisgridminor;
  haImag.XMinorGrid = app.axisgridminor;
  
  % set label
  set(app.tUnits, 'String', newax.xlabel);
  
  % put axis menu, select View
  if app.PlotType ~= 5, set(haReal, 'View',[0 90]); set(haImag, 'View',[0 90]); end
  axtoolbar(haReal, 'default');
  
catch err, ShowError(err);
end
% --------------------------------------------------------------------
function mViewAxisSet(h, ~, app, varargin)
hdl = [app.mViewAxistight, app.mViewAxisauto app.mViewAxisfixed];
hdl1 = [app.mViewAxissquare app.mViewAxisnormal];
app = guidata(app.MainFigure);
switch h
  case app.mViewAxistight
    axis(app.haReal, 'tight')
    axis(app.haImag, 'tight')
    app.axisset = 'tight';
  case app.mViewAxisauto
    axis(app.haReal, 'auto')
    axis(app.haImag, 'auto')
    app.axisset = 'auto';
  case app.mViewAxisfixed
    app.axisset = 'auto';
  case app.mViewAxisnormal
    axis(app.haReal, 'normal')
    axis(app.haImag, 'normal')
    app.axisaspect = 'normal';
    set(h,'Checked','on');
    set(hdl1(hdl1~=h),'Checked','off');
    guidata(app.MainFigure, app);
    return
  case app.mViewAxissquare
    axis(app.haReal, 'square')
    axis(app.haImag, 'square')
    app.axisaspect = 'square';
    set(h,'Checked','on');
    set(hdl1(hdl1~=h),'Checked','off');
    guidata(app.MainFigure, app);
    return
  case app.mViewAxisEqual
    if strcmp(get(h,'Checked'),'on')
      app.axisaspect1 = 'normal';
      set(h,'Checked','off')
    else
      app.axisaspect1 = 'equal';
      set(h,'Checked','on')
    end
    axis(app.haReal, app.axisaspect1)
    axis(app.haImag, app.axisaspect1)
    guidata(app.MainFigure, app);
    return
  case app.mViewShowGrid
    if strcmp(app.axisgrid,'on'), app.axisgrid = 'off';
    else, app.axisgrid = 'on';
    end
    grid(app.haReal, app.axisgrid)
    grid(app.haImag, app.axisgrid)
    guidata(app.MainFigure, app);
    set(h,'Checked',app.axisgrid);
    return
  case app.mViewShowGridMinor
    if strcmp(app.axisgridminor,'on'), app.axisgridminor = 'off';
    else, app.axisgridminor = 'on';
    end
    app.haReal.XMinorGrid = app.axisgridminor;
    app.haImag.XMinorGrid = app.axisgridminor;
    guidata(app.MainFigure, app);
    set(h,'Checked',app.axisgridminor);
    return
end
set(h,'Checked','on');
set(hdl(hdl~=h),'Checked','off');
guidata(app.MainFigure, app);

% --------------------------------------------------------------------
function DataSource_Callback(h, ~, app, varargin)

switch h
  case {app.rbSource, app.mViewSource}
    SetDataSource(1, app)
  case {app.rbOutput, app.mViewOutput}
    SetDataSource(2, app)
  otherwise
    SetDataSource(3, app)
end

% --------------------------------------------------------------------
function SetDataSource(source, app)
switch source
  case 1
    set(app.rbOutput,'Value',0);
    set(app.rbSource,'Value',1);
    set(app.mViewOutput,'Checked','off');
    set(app.mViewSource,'Checked','on');
    set(app.mViewSnO,'Checked','off');
    app.DataSource = 1;
  case 2
    set(app.rbOutput,'Value',1);
    set(app.rbSource,'Value',0);
    set(app.mViewOutput,'Checked','on');
    set(app.mViewSource,'Checked','off');
    set(app.mViewSnO,'Checked','off');
    app.DataSource = 2;
  case 3
    set(app.rbOutput,'Value',1);
    set(app.rbSource,'Value',1);
    set(app.mViewOutput,'Checked','on');
    set(app.mViewSource,'Checked','on');
    set(app.mViewSnO,'Checked','on');
    app.DataSource = 3;
  otherwise
    set(app.rbOutput,'Value',0);
    set(app.rbSource,'Value',0);
    set(app.mViewOutput,'Checked','off');
    set(app.mViewSource,'Checked','off');
    app.DataSource = 3;
end
guidata(app.MainFigure, app);
UpdateSlider(app);

% --------------------------------------------------------------------
function DataPart_Callback(h, ~, app, varargin)
switch h
  case {app.bReal, app.mViewReal}
    SetDataPart(1, app);
  case {app.bImag, app.mViewImag}
    SetDataPart(2, app);
  otherwise
    SetDataPart(3, app);
end

% --------------------------------------------------------------------
function SetDataPart(type, app)
hdl = [app.bReal, app.bImag, app.rbMagnitude];
hdl1 = [app.mViewReal, app.mViewImag, app.mViewMagnitude];
switch type
  case 1
    set(app.bReal,'Value',1);
    set(hdl(hdl ~= app.bReal),'Value',0);
    set(app.mViewReal,'Checked','on');
    set(hdl1(hdl1~=app.mViewReal),'Checked','off');
    app.DataPart = 1;
  case 2
    set(app.bImag,'Value',1);
    set(hdl(hdl ~= app.bImag),'Value',0);
    set(app.mViewImag,'Checked','on');
    set(hdl1(hdl1~=app.mViewImag),'Checked','off');
    app.DataPart = 2;
  otherwise
    set(app.rbMagnitude,'Value',1);
    set(hdl(hdl ~= app.rbMagnitude),'Value',0);
    set(app.mViewMagnitude,'Checked','on');
    set(hdl1(hdl1~=app.mViewMagnitude),'Checked','off');
    app.DataPart = 3;
end

guidata(app.MainFigure, app);
MainFigure_ResizeFcn(0, 0, app);
PlotData(app.MainFigure);

% --------------------------------------------------------------------
function SetPlotType(app, plot_type)
hdl = [app.tb1D,  app.tbStacked, app.tbContour, app.tbDensity];

switch plot_type
  case 1
    set(app.tb1D,'Value',1);
    set(hdl(hdl ~= app.tb1D),'Value',0);
    app.PlotType = 1;
    isScroller = true;
  case 2
    set(app.tbStacked,'Value',1);
    set(hdl(hdl ~= app.tbStacked),'Value',0);
    app.PlotType = 2;
    isScroller = false;
  case 3
    set(app.tbContour,'Value',1);
    set(hdl(hdl ~= app.tbContour),'Value',0);
    app.PlotType = 3;
    isScroller = false;
  case 4
    set(app.tbDensity,'Value',1);
    set(hdl(hdl ~= app.tbDensity),'Value',0);
    app.PlotType = 4;
    isScroller = false;
end

if isScroller ~= app.pb2DSelect.Visible
  str = {'off', 'on'};
  MainFigure_ResizeFcn([], [], app);
  app.pb2DSelect.Visible = str{isScroller + 1};
  app.sl2Dplot.Visible = str{isScroller + 1};
end
app.isScroller = isScroller; 

% incorrect ...
if app.isScroller
  dim = size(app.src.y);
  if dim(2) > 1
    set(app.pb2DSelect, 'String', 'X');
    set(app.sl2Dplot, 'Min', 1, 'Max', dim(2), 'Value', 1,...
      'SliderStep', [1 10]*1/dim(2));
    app = sl2Dplot_Callback(app.sl2Dplot, [], app);
  end
else
  Set(app, 'cursor', '');
end

guidata(app.MainFigure, app);
PlotData(app.MainFigure);
% --------------------------------------------------------------------

function app = ddDataTypeValueChanged(~, ~, app, varargin)
app.LoadScriptOptions = app.ScriptLoad('options', [ddValue(app.ddExt), ddValue(app.ddDataType)]);
sel = get(app.ddDataType, 'Value');
if sel < 1
  return
end
str = get(app.ddDataType, 'String');
app.Datatype = str{sel};
app = DataLoadOptions(app);
guidata(app.MainFigure, app);

% --------------------------------------------------------------------
function app = LoadDirectory(app, dir_path)
last_pwd = pwd;
if ispc
  if size(dir_path,2) > 0
    if sum(dir_path == ':')==0
      try cd (fullfile(app.dir_path, dir_path)); catch err, ShowError(err); end
    else
      cd (dir_path);
    end
  else
    try cd (app.dir_path); catch err, ShowError(err); end
  end
else
  try cd (fullfile(app.dir_path, dir_path)); catch err, ShowError(err); end
end
app.dir_path = pwd;
dir_dir_struct = dir(app.dir_path);
dir_dir_struct = dir_dir_struct([dir_dir_struct.isdir]);
if sum(findstr([dir_dir_struct.name], '.'))==6
  dir_dir_struct = dir_dir_struct(3:end);
end
[sorted_dirnames,sorted_dirindex] = sortrows({dir_dir_struct.name}');

dir_size = size(dir_dir_struct, 1);

if ispc
  dir_struct = dir(fullfile(app.dir_path, app.ext));
else
  dir_struct = dir(fullfile(app.dir_path, lower(app.ext)));
  dir_struct_tmp = dir(fullfile(app.dir_path, upper(app.ext)));
  for ii=1:length(dir_struct_tmp)
    dir_struct(end+1,1)=dir_struct_tmp(ii);
  end
end
dir_struct = dir_struct(~[dir_struct.isdir]);
file_size = size(dir_struct, 1);

[sorted_names,sorted_index] = sortrows({dir_struct.name}');
sorted_index = sorted_index+dir_size;

% final_names =
final_index = [sorted_dirindex; sorted_index];
final_names = {};
for ii=1:dir_size
  final_names{ii} = ['[' sorted_dirnames{ii} ']'];
end
for ii=1:file_size
  final_names{ii+dir_size} = sorted_names{ii};
end

app.file_names = final_names;
app.is_dir = [ones(1, dir_size) dir_struct.isdir];
app.sorted_index = [final_index];
set(app.lDirlist,'String',app.file_names,'Value',1)
set(app.ePath,'String',pwd, 'TooltipString', pwd)
app.ini.KazanViewer.last_extension = ddValue(app.ddExt);
app.ini.KazanViewer.last_path = app.ePath.String;
guidata(app.MainFigure,app);

cd(last_pwd);

% --------------------------------------------------------------------
function lDirlist_Callback(~, ~, app, varargin)

index_selected = get(app.lDirlist,'Value');
file_list = get(app.lDirlist,'String');
if isempty(index_selected), return; end
filename = file_list{index_selected}; % Item selected in list box
if app.is_dir(app.sorted_index(index_selected)) % If directory
  %strip directory brackets
  filename = filename(2:end-1);
  % Load list box with new directory when double clicked
  seltype = get(app.MainFigure, 'SelectionType');
  if strcmp(seltype, 'open'), app = LoadDirectory(app, filename); end
else
  app.src.ax = [];
  app.src.y = [];
  
  if 1 == 1
    SetDataSource(1, app);
    app = guidata(app.MainFigure);
  end
  
  app=LoadFile(fullfile(app.dir_path, filename), app);
  
  % call update and app repair for all plugins
  MakePlugins(app, 'onload', []);
  guidata(app.MainFigure,app);
  
  app = guidata(app.MainFigure);
  
  source = app.DataSource;
  guidata(app.MainFigure, app);
  %   if source ~=1, source = 3; end;
  SetDataSource(source, app);
end

% --------------------------------------------------------------------
function eDirectory_Callback(~, ~, app, varargin)
d = app.ePath.String;
if size(d, 2)==1
  d = [d ':\'];
end
LoadDirectory(app, d);

% --------------------------------------------------------------------
function pbUp_Callback(~, ~, app, varargin)
A = split(app.dir_path, '\');
last_dir = ['[' ,A{end}, ']'];
LoadDirectory(app, '..');

dirs = get(app.lDirlist, 'String');
whichone = strcmp(dirs, last_dir);
k = find(whichone);
set(app.lDirlist, 'Value', k);

% --------------------------------------------------------------------
function app = LoadFile(file_name, app)
fileinfo = dir(file_name);
if fileinfo.bytes >= 2^20
  button = questdlg('Do you want to continue reading?',...
    ['File size ' num2str(fileinfo.bytes/2^20, '%10.2f'), 'Mb'],...
    'Yes','No','Yes');
  if strcmp(button, 'No'), return; end
end

app.LastLoadingScript = {};
if get(app.tb1D, 'Value')
  cmContexMenu2(app.tb1D, [], app);
end
app.fname = file_name;

options = cell(3,1);
if length(app.LoadScriptOptions) >= 1
  style = get(app.ReadOptHandle(1), 'Style');
  switch style
    case 'popupmenu'
      idx = get(app.ReadOptHandle(1), 'Value');
      items = app.LoadScriptOptions{1}.Data;
      options{1} = items{idx};
  end
end
if length(app.LoadScriptOptions) >= 2
  style = get(app.ReadOptHandle(2), 'Style');
  switch style
    case 'popupmenu'
      idx = get(app.ReadOptHandle(2), 'Value');
      items = app.LoadScriptOptions{2}.Data;
      options{2} = items{idx};
  end
end
if length(app.LoadScriptOptions) >= 3
  style = get(app.ReadOptHandle(3), 'Style');
  switch style
    case 'edit'
      options{3} = get(app.ReadOptHandle(3), 'String');
  end
end
app.LastLoadingScript = app.ScriptLoad('script', [ddValue(app.ddExt), ddValue(app.ddDataType)], file_name, options);

try
  % load script:
  for k = 1:size(app.LastLoadingScript, 2)
    eval(app.LastLoadingScript{k});
  end
  app.LastLoadingScript{end+1} = '% plot(ax.x,real(y),ax.x,imag(y))';
catch err
  ShowError(err);
  error(['LoadFile: Data load error of type ',app.Datatype, '.']);
end

% this will convert data into 2 dimensional variant
sz = size(y);
if size(sz,2) > 2
  warning('LoadFile: Data dimension is reduced.');
  y = reshape(y, [sz(1), prod(sz(2:end))]);
  ax.y = (1:prod(sz(2:end)))';
  sz = size(y);
end

% Convert DSC parameters in AX parameters
ax = app.ParameterReader('load', ax, dsc);

app.src.y   = y;
app.src.ax  = ax;
app.src.dsc = dsc;
app.src.filename = file_name;

app.src.ax.ext = app.ext;
app.src.ax.Datatype = app.Datatype;
app.src.ax.filt = app.filter;
app.src.ax.diff = app.process;

app.src = Verify(app, app.src);

% load parameters
app = MakeParameters(app, 'load', app.src.ax);
[app, app.src.ax] = MakeParameters(app, 'from_editor', app.src.ax);

SetPlotType(app, app.PlotType);
app = guidata(app.MainFigure);
UpdateSlider(app);

if isfield(app.src.ax, 'title')
  set(app.MainFigure, 'Name', ['KAZAN - ',file_name, ' [',app.src.ax.title,']'])
else
  set(app.MainFigure, 'Name', ['KAZAN - ',file_name, ' [',app.process,']'])
end

% --------------------------------------------------------
function mTools(h, ~, app, varargin)
switch h
  case app.mToolsClearSource
    app.src.y=[];
    app.src.ax.x=[];
    app.src.ax.y=[];
    app.src.dsc=[];
    guidata(app.MainFigure, app);
    PlotData(app.MainFigure);
    return;
  case app.mToolsFixProcessing
    app.src.ax.filter   = safeget(app,'filter','');
    app.src.ax.filtpar1 = safeget(app,'filterpar1','auto');
    app.src.y=processing(app.src.y,app.src.ax);
    app=cmContexMenu1(app.cmNormal,[],app);
    app=cmContexMenu2(app.cmxy,[],app);
    app=Filter_CallBack(app.cmNofilter,[],app);
    guidata(app.MainFigure, app);
    PlotData(app.MainFigure);
    return;
  case app.mToolsFixSlice
    if app.PlotType == 1
      app.src = GetSlice(app.src,app.Projection,app.Selection);
      app.Projection = 1;
      app.AntiProjection = 2;
      guidata(app.MainFigure, app);
      UpdateSlider(app);
    end
    return;
  case app.mToolsLoadscript
    f = '';
    dim = size(app.LastLoadingScript, 2);
    for i=1:dim, f = [f, newline, app.LastLoadingScript{1,i}]; end
    disp('Data load script is copied to the clipboard.');
    clipboard('copy',f)
    disp(f);
    return;
  case app.mToolsExecLoadscript
    dim = size(app.LastLoadingScript, 2);
    for i=1:dim
      disp(app.LastLoadingScript{1,i});
      evalin('base', app.LastLoadingScript{1,i}, '');
    end
    return;
  case app.mToolsZoom
    state = get(app.mToolsZoom, 'Checked');
    if strcmp('off', state)
      val = 'on';
      P(1:16, 1:16) = NaN;
      P([1, 2, 4:16], 3)= 1;
      P(3, [1, 2, 4:16])= 1;
      P([4, 14], 7:11) = 1;
      P(7:11, [4, 14]) = 1;
      P([6, 12], [5, 13]) = 1;
      P([5, 13], [6, 12]) = 1;
      P(13, 13) = 1;
      P([13, 14], [14, 13]) = 1;
      P([15, 16], [16, 15]) = 1;
      P(7, 11) = 1;
      P([8, 9, 10], 12) = 1;
      P(11, 11) = 1;
      set(app.MainFigure,'Pointer','custom','PointerShapeCData',P,...
        'PointerShapeHotSpot', [3, 3]);
    else
      val = 'off';
      set(app.MainFigure,'Pointer','arrow');
    end
    set(app.mToolsZoom, 'Checked', val);
    return;
  case app.mToolsAutoZoomOff
    str = {'off', 'on'};
    stat = get(app.mToolsAutoZoomOff, 'Checked');
    idx = find(strcmp(str, stat));
    set(app.mToolsAutoZoomOff, 'Checked', str{~(idx-1) + 1});
    app.autoZoomOff = str{~(idx-1) + 1};
    guidata(app.MainFigure, app);
    return;
  case app.mToolsZoomFixed
    str = {'off', 'on'};
    stat = get(app.mToolsZoomFixed, 'Checked');
    idx = find(strcmp(str, stat));
    set(app.mToolsZoomFixed, 'Checked', str{~(idx-1) + 1});
    app.FixedZoomOpt = str{~(idx-1) + 1};
    guidata(app.MainFigure, app);
    return;
  case {app.mCoord, app.tbMeasure}
    p=[];
    disp('press any key to cancel')
    while(1)
      [x,y, button]=ginput(1);
      if isempty(button), button=2; end
      switch button
        case 1
          if app.DataSource==2
            newax = getplotaxis(app.xrepresent, app.yrepresent, app.out{1}.ax);
          else
            newax = getplotaxis(app.xrepresent, app.yrepresent, app.src.ax);
          end
          if app.Projection == 1 || app.PlotType > 2, xx = newax.x; yy = newax.y; else, xx = newax.y; yy = newax.x; end
          
          [~,xpnt] = min(abs(xx-x));
          if app.PlotType < 3
            s = sprintf('[%d] x= %g, y= %g',xpnt, x,y);
          else
            [~,ypnt] = min(abs(yy-y));
            s = sprintf('[%d,%d] x= %f, y= %f', xpnt, ypnt, x,y);
          end
          Set(app, 'cursor',s);
          disp(s)
          p(end+1, :) = [x,y];
        otherwise
          break;
      end
    end
    switch size(p,1)
      case 2
        % distance
        s = sprintf('dist x= %g,    y= %f',p(2,1)-p(1,1),p(2,2)-p(1,2));
        disp(s)
        Set(app, 'message', s);
        fprintf('center x= %g,    y= %f\n',(p(2,1)+p(1,1))/2,(p(2,2)+p(1,2))/2)
      case 4
        % amplitude relation
        s = sprintf('(y4-y3)/(y2-y1) = %f',(p(4,2)-p(3,2))/(p(2,2)-p(1,2)));
        Set(app, 'message', s);
        fprintf('(x4-x3)/(x2-x1) = %f\n',(p(4,1)-p(3,1))/(p(2,1)-p(1,1)));
        disp(s)
    end
    disp(newline)
end
return;

% --------------------------------------------------------------------
function app = mFileHistory(h, eventdata, app, varargin)
menus = get(app.mFile, 'Children');
firstpos = get(app.mFileNextHistory,'Position')+1;
lastpos = size(menus, 1);
selected = 1;

switch(h)
  case app.mFileAddHistory
    thisfile = findobj(menus, 'Label', app.fname);
    if ~isempty(app.fname) && isempty(thisfile)
      app.fhistory{end+1}.fname = app.fname;
      app.fhistory{end}.ax = app.src.ax;
    end
    selected = size(app.fhistory, 2);
  case app.mFileRemoveHistory
    selmenus = findobj(menus, 'Checked', 'on');
    if isempty(selmenus), return; end
    delfile = get(selmenus, 'Label');
    for k = 1:size(app.fhistory, 2)
      if app.fhistory{k}==delfile
        app.fhistory = {app.fhistory{[1:(k-1),(k+1):end]}};
        break;
      end
    end
    selected = 0;
  case app.mFileRemoveAllHistory
    app.fhistory = {};
  case app.mFileNextHistory
    if size(menus, 1) < firstpos, return; end
    selmenus = findobj(menus, 'Checked', 'on');
    if isempty(selmenus)
      pos = firstpos;
    else
      pos = get(selmenus, 'Position');
      if pos == lastpos, pos = firstpos; else, pos = pos+1; end
    end
    nextmenu = findobj(menus, 'Position', pos);
    FileHistory(nextmenu, eventdata, app, varargin);
    return
  case app.mFilePreviousHistory
    if size(menus, 1) < firstpos, return;  end
    selmenus = findobj(menus, 'Checked', 'on');
    if isempty(selmenus)
      pos = firstpos;
    else
      pos = get(selmenus, 'Position');
      if pos == firstpos, pos = lastpos; else, pos = pos-1; end
    end
    nextmenu = findobj(menus, 'Position', pos);
    FileHistory(nextmenu, eventdata, app, varargin);
    return;
end

% menu recreating
delete(findobj(menus, 'UserData', 'history'));

for k=1:size(app.fhistory, 2)
  if k == 1, sep='on'; else, sep = 'off'; end
  if k == selected, sel='on'; else, sel = 'off'; end
  uimenu(app.mFile, 'Label',app.fhistory{k}.fname, 'UserData', 'history',...
    'Tag', 'mPluginsReposition', 'Separator', sep, 'Checked', sel,...
    'Callback', 'kazan(''FileHistory'',gcbo,[],guidata(gcbo))');
end
guidata(h, app)

% --------------------------------------------------------------------
function FileHistory(h, eventdata, app, varargin)
menus = get(app.mFile, 'Children');
set(menus, 'Checked', 'off');
[fpath,name,ext] = fileparts(get(h, 'Label'));
app.dir_path = fpath;
firstpos = get(app.mFileNextHistory,'Position')+1;
pos = get(h,'Position') - firstpos + 1;

% set old extension
str = get(app.ddExt, 'String');
whichext = strcmpi(str, app.fhistory{pos}.ax.ext);
set(app.ddExt, 'Value', find(whichext));
app = ddExtValueChanged(gcbo, [], app, varargin);

str = get(app.ddDataType, 'String');
if ~strcmpi(str{get(app.ddDataType, 'Value')}, app.fhistory{pos}.ax.Datatype)
  % set old datatype
  whictype = strcmpi(str, app.fhistory{pos}.ax.Datatype);
  set(app.ddDataType, 'Value', find(whictype));
  app = ddDataTypeValueChanged(gcbo, [], app, varargin);
end

% set old filename
str = get(app.lDirlist, 'String');
whichfile = strcmp(str, [name, ext]);
set(h, 'Checked', 'on');
set(app.lDirlist, 'Value', find(whichfile));
lDirlist_Callback(app.lDirlist, eventdata, app, varargin)

% --------------------------------------------------------------------
function app = cmContexMenu1(h, ~, app, varargin)
opt = [app.cmNormal,app.cmDiff1,app.cmIntegrate];
switch h
  case app.cmDiff1
    app.process = 'diff';
  case app.cmIntegrate
    app.process = 'integr';
  otherwise
    app.process = '';
end
set(opt, 'Checked', 'off');
set(h, 'Checked', 'on');
app.src.ax.diff = app.process;
guidata(h,app);
PlotData(app.MainFigure);
%lDirlist_Callback(app.lDirlist, eventdata, app, varargin);

% --------------------------------------------------------------------
function app = cmContexMenu2(h, ~, app, varargin)
xopt = [app.cmxLinear, app.cmxPoints, app.cmReverseX, app.cmSemilogX, app.cmGspace, ...
  app.cmxPPM, app.cmxPPMmin1];
yopt = [app.cmyLinear, app.cmyPoints, app.cmReverseY, app.cmSemilogY ...
  app.cmyPPM, app.cmyPPMmin1];
if h==app.cmxy
  set(xopt, 'Checked', 'off');
  set(yopt, 'Checked', 'off');
  app.xrepresent = 'plot';
  app.yrepresent = 'plot';
  set(app.cmxLinear, 'Checked', 'on');
  set(app.cmyLinear, 'Checked', 'on');
elseif sum(h==xopt)
  set(xopt, 'Checked', 'off');
  switch h
    case app.cmReverseX
      app.xrepresent = 'reverse';
    case app.cmSemilogX
      app.xrepresent = 'semilog';
    case app.cmGspace
      app.xrepresent = 'gspace';
    case app.cmxPPM
      app.xrepresent = 'ppm';
    case app.cmxPPMmin1
      app.xrepresent = 'ppm-1';
    case app.cmxPoints
      app.xrepresent = 'pnt';
    otherwise
      app.xrepresent = 'plot';
  end
  set(h, 'Checked', 'on');
elseif sum(h==yopt)
  set(yopt, 'Checked', 'off');
  switch h
    case app.cmReverseY
      app.yrepresent = 'reverse';
    case app.cmSemilogY
      app.yrepresent = 'semilog';
    case app.cmyPPM
      app.yrepresent = 'ppm';
    case app.cmyPPMmin1
      app.yrepresent = 'ppm-1';
    case app.cmyPoints
      app.yrepresent = 'pnt';
    otherwise
      app.represent = 'plot';
  end
  set(h, 'Checked', 'on');
end
guidata(h,app);
PlotData(app.MainFigure);

% --------------------------------------------------------------------
function varargout = Filter_CallBack(h, ~, app, varargin)
opt  = [app.cmNofilter, app.cmRCfilter, app.cmMvgaver, app.cmSavGol];
opt1 = [app.cmFiltAuto, app.cmFilt1D, app.cmFilt2D, app.cmFilt1D2D];
if sum(h==opt)
  set(opt, 'Checked', 'off');
  switch h
    case app.cmRCfilter
      app.filter = 'rc';
    case app.cmMvgaver
      app.filter = 'mvgaver';
    case app.cmSavGol
      app.filter = 'savgol';
    otherwise
      app.filter = '';
  end
  set(h, 'Checked', 'on');
  app.src.ax.filt = app.filter;
elseif sum(h==opt1)
  set(opt1, 'Checked', 'off');
  switch h
    case app.cmFilt1D
      app.filterpar1 = '1D';
    case app.cmFilt2D
      app.filterpar1 = '2D';
    case app.cmFilt1D2D
      app.filterpar1 = '1D2D';
    otherwise
      app.filterpar1 = 'auto';
  end
  set(h, 'Checked', 'on');
  app.src.ax.filtpar1 = app.filterpar1;
end
guidata(h,app);
PlotData(app.MainFigure);
if nargout > 0, varargout{1}=app; end

% --------------------------------------------------------------------
function mHelp(h, ~, app)

switch h
  case app.mHelp1
    msgbox({'Parameters(fields) that affect visualization:'; ...
      'title     - data title'; ...
      'x/y       - 1D axis array (column!)   REQUIRED'; ...
      'x/ylabel  - axis labels'; ...
      'Color     - color of the trace b/g/r/c/m/y/k or [r g b], rgb=0..1';...
      'LineStyle - style of the line -/:/-./--'; ...
      'LineWidth - width of the line 0..inf'; ...
      'Marker    - style of dots none/./o/x/+/*/s/d/v/^/</>/p/h'; ...
      'contour   - for 2D plots - contour level';''; ...
      'Plot type related:'; ...
      'reference - for PPM plot'; ...
      'freq1     - for g-plot';''; ...
      'Data processing:'; ...
      'dx, s          - shift of abscissa/ordinate(2D)'; ...
      'diff, ps_mod    - pseudo modulation parameters diff/integr'; ...
      'filt, tc, pol   - filtering parameters rc/mvgaver/savgol, 1..inf, 1..inf';''; ...
      'Fields of ax structure are accessible throught the';...
      'combobox and edit controls at the bottom of the main window'},...
      'Axis Data Structure (ax)', 'none')
  case app.mHelp2
    msgbox({'Press a) middle button or b) left and right'; ...
      'mouse buttons at the same time or ';...
      'c) SHIFT and left mouse button';...
      'to trace the mouse cursor position';'';...
      'In measurements mode:';...
      '2 clicks followed by ENTER - distance';...
      '4 clicks followed by ENTER - ratio'}, 'Mouse Cursor', 'none')
  case app.mHelp3
    msgbox({'1: Load data';...
      '    Select file extension';'    Select file type';...
      '    Change loading parameters if needed';...
      '    Find directory';'    Select file';'';...
      '2: Adjust the view';...
      '    Right click on the plot, select options';'';...
      '3: Process the data';...
      '    Run plugin from Plugins menu';...
      '    Do required procesing';'';...
      '4: Copy processed data from Output to Source (<<)';...
      '   Repeat processing if needed';'';...
      '5: Print plot';...
      '   select Tools/Show in Window'},...
      'How To Start', 'none')
  case app.mHelp4
    msgbox({'All data-loading routines have similar structure:'; ...
      '[ax,y,dsc] = routine(filename,...),';...
      'where ''ax'' is the axis data structure (see corresponding help),'; ...
      '''y'' is a 2D data array (first dimension corresponds to ''ax.x''),';...
      '''dsc'' is the structure with arbitrary field names (information)'},...
      'Data Load Routines', 'none')
  case app.mHelp5
    msgbox({['Plugins are unified data processing routines, which take data ',...
      'from Kazan Viewer and return result to Viewer. The origin of data ',...
      'is called ''Source'' the result is sent to ''Output''. Plugins can be ',...
      'applied in a serial way by transfering of the output of precursed plugin ',...
      'to the Source (button ''<<'' of the Viewer.)'];'';...
      ['Plugins of Kazan Viewer are the figures (*.fig) located in the ',...
      'directory of the viewer. After the loading of a plugin ',...
      'the Viewer set the ''app.hh'' field of plugin to the ',...
      'Viewer''s handle . The handle structure of Viewer ',...
      'can then be obtained by ''ha = guidata(app.hh)'' call.'];'';...
      'The Source data located at ''ha.src''';...
      'The Output data can be stored to ''ha.out{}''';...
      'The ''src'' and ''out'' structures have ''y'', ''ax'' and ''dsc'' fields.';...
      'See Data Load Routines for description.';'';...
      'Function calls:';...
      'The plotting function of plugin have to be finished by:';...
      '    guidata(app.hh, ha);';...
      '    [path,name,ext,ver] = fileparts(get(h.hh, ''FileName''));';...
      '    eval([name ''(''''SetDataSource'''', 2, hand)'']);';...
      'When data are loaded or plugin is opened';...
      'the ''FileUpdate(app)'' routine of plugin is called.';'';...
      'Ini file:';...
      'Ini file (kazan.ini) can be accessed using ''inimanage'' routine.'},...
      'Plugins Programming', 'none')
  case app.mHelp6
    msgbox({['Select ''Tools/Show in Separate Figure'' menu.',...
      'Then the standard Matlab figure will appear.'];...
      'Optionally the plot can be conditioned using Export Image utility';...
      'or Profile Manager (see corresponding menus)';'';...
      'To create multiple axis on the figure use Show in Axis menu.';...
      ''},...
      'How to print', 'none')
  otherwise, msgbox('No help found');
end
% --------------------------------------------------------------------
function res = superplot(xrepresent, yrepresent, ax, y, p)
c = safeget(ax, 'Color', 'b');
l = safeget(ax, 'LineStyle', '-');
lw = safeget(ax, 'LineWidth', 1);
m = safeget(ax, 'Marker', 'none');

res = plot(ax.x, y, 'Color', c, 'LineStyle', l, 'LineWidth', lw, 'Marker', m, 'Parent', p);

if strcmp(xrepresent, 'semilog'), set(p, 'XScale','log', 'XDir', 'normal');
else
  set(p, 'XScale','linear');
  if strcmp(xrepresent, 'reverse') || strcmp(xrepresent, 'gspace')
    set(p, 'XDir','reverse');
  else, set(p, 'XDir','normal');
  end
end
if strcmp(yrepresent, 'semilog'), set(p, 'YScale','log', 'YDir', 'normal');
else
  set(p, 'YScale','linear');
  if strcmp(yrepresent, 'reverse'), set(p, 'YDir','reverse');
  else, set(p, 'YDir','normal');
  end
end

% --------------------------------------------------------------------
function eParameterValueChanged(~, ~, app, varargin)
[app, app.src.ax] = MakeParameters(app, 'from_editor', app.src.ax);
guidata(app.MainFigure, app);
PlotData(app.MainFigure);

% --------------------------------------------------------------------
function ddSelParameterValueChanged(~, ~, app, varargin)
MakeParameters(app, 'update_editor');

% --------------------------------------------------------------------
function mFile2Source(~, ~, app, varargin)
switch app.DataSource
  case 1
  case {2, 3}
    switch length(app.out)
      case 0, n=0;
      case 1, n=1;
      otherwise
        cols = {}; titles = {};
        for k = 1:size(app.out, 2)
          cols{end+1, 1} =  safeget(app.out{k}.ax, 'Color', [0;0;1]);
          titles{end+1, 1} = safeget(app.out{k}, 'title', '');
        end
        n = kv_seltrace(cols, titles);
    end
    if n
      app.src = app.out{n};
      SetDataSource(1, app);
    end
end

% --------------------------------------------------------------------
function app = sl2Dplot_Callback(h, ~, app, varargin)
app.Selection = floor(get(app.sl2Dplot, 'Value')+0.5);
guidata(h, app);
str = {'X', 'Y'};
if app.DataSource == 1
  data = safeget(app.src.ax, lower(str{app.AntiProjection}), []);
  label = safeget(app.src.ax, [lower(str{app.AntiProjection}), 'label'], '?');
else
  data = safeget(app.out{1}.ax, lower(str{app.AntiProjection}), []);
  label = safeget(app.out{1}.ax, [lower(str{app.AntiProjection}), 'label'], '?');
end
if isempty(data)
  str1 = '';
else
  data = data(min(app.Selection,length(data)));
  str1 = [str{app.AntiProjection}, ' (', label,'): ' num2str(data), ' [', num2str(app.Selection),']'];
end

Set(app, 'cursor', str1);
PlotData(app.MainFigure);

% call update and app repair for all plugins

par.Selection = app.Selection;
par.Dim = lower(str{app.AntiProjection});
for k=1:length(app.Plugins)
  if contains(class(app.Plugins{k}), 'Figure')
    try
      [~,name] = fileparts(get(app.Plugins{k}, 'FileName'));
      str = [name, '(''SelectionUpdate'',app.Plugins{k}, par, guidata(app.Plugins{k}));'];
      %             try eval(str); catch; end
    catch  err
      ShowError(err);
      DeletePlugins(app.Plugins{k}, app.MainFigure);
    end
  end
end

% --------------------------------------------------------------------
function pb2DSelect_Callback(h, ~, app, varargin)
str = {'X', 'Y'};
app.Projection = safeget(app,'Projection',1);
if app.Projection == 1, app.Projection = 2; app.AntiProjection = 1;
else, app.Projection = 1; app.AntiProjection = 2;
end
set(h, 'String', str{app.Projection});
guidata(h, app);
UpdateSlider(app);
% --------------------------------------------------------------------

function mFileWorkspace(h, ~, app, varargin)
if h ~= app.mFileSaveWorksp
  tt = evalin('base', 'tmp');
  disp('variable ''tmp'' loaded from workspace');
  app.src = VerifyData(tt);
  guidata(h, app);
  PlotData(app.MainFigure);
  if isfield(app.src.ax, 'title')
    set(app.MainFigure, 'Name', ['KAZAN - workspace [',app.src.ax.title,']'])
  else
    set(app.MainFigure, 'Name', ['KAZAN - workspace [',app.process,']'])
  end
  
  
  MakePlugins(app, 'onload', []);
else
  assignin('base', 'tmp', app.src);
  disp('variable ''tmp'' created');
  disp(app.src);
end
% --------------------------------------------------------------------
function Zoom_Callback(h, ~, app, varargin)

lim = get(app.haReal, 'XLim');
dx  = lim(1)-lim(2);
limy = get(app.haReal, 'YLim');
dy  = limy(1)-limy(2);

switch h
  case app.tbX_half, lim = lim + dx * [1, -1] / 2;
  case app.tbX_twice, lim = lim - dx * [1, -1] / 4;
  case app.tbY_half, limy = limy + dy * [1, -1] / 2;
  case app.tbY_twice, limy = limy - dy * [1, -1] / 4;
end
set(app.haReal, 'XLim', lim, 'YLim', limy);
set(app.haImag, 'XLim', lim);

% --------------------------------------------------------------------
function mViewRange(~, ~, app, varargin)
viewarea_func(app)

% --------------------------------------------------------------------
function newax = getplotaxis(xfname, yfname, ax)
newax = ax;

dx = safeget(ax, 'dx', 0); newax.dx=0;
dy = safeget(ax, 'dy', 0); newax.dy=0;
gref = safeget(ax, 'freq1', 34E9)*7.144773452e-11*1e4; % planck/bmagn
if gref == 0.0, gref = 1.; end
ref = safeget(ax, 'reference', 34E9);
if ref == 0.0, ref = 1.; end

xlabel = safeget(ax, 'xlabel', '');
pos = strfind(xlabel, ',');
if isempty(pos), xvar='?,'; else, xvar = xlabel(1:min(pos)); end
ylabel = safeget(ax, 'ylabel', '');
pos = strfind(ylabel, ',');
if isempty(pos), yvar='?,'; else, yvar = ylabel(1:min(pos)); end

if contains(xfname, 'gspace')
  newax.x = gref./(ax.x+dx);
  newax.xlabel = 'g-factor,';
  newax.unit = '';
elseif contains(xfname, 'ppm-1')
  newax.x = ((ax.x+dx)./ref - 1)*1E6;
  newax.xlabel = [xlabel,'ppm'];
  newax.unit = 'ppm';
elseif contains(xfname, 'ppm')
  newax.x = (ax.x+dx)./ref*1E6;
  newax.xlabel = [xvar,'ppm'];
  newax.unit = 'ppm';
elseif contains(xfname, 'pnt')
  newax.x = (1:size(ax.x,1)).';
  newax.xlabel = [xvar,'pts'];
  newax.unit = 'pts';
else
  newax.x = ax.x+dx;
  newax.xlabel = xlabel;
end

if contains(yfname, 'gspace')
  newax.y = gref./(ax.y+dy);
  newax.ylabel = 'g-factor,';
elseif contains(yfname, 'ppm-1')
  newax.y = ((ax.y+dy)./ref - 1)*1E6;
  newax.ylabel = [yvar,'ppm'];
elseif contains(yfname, 'ppm')
  newax.y = (ax.y+dy)./ref*1E6;
  newax.ylabel = [yvar,'ppm'];
elseif contains(yfname, 'pnt')
  newax.y = (1:size(ax.y,1))';
  newax.ylabel = [yvar,'pts'];
else
  newax.y = ax.y+dy;
  newax.ylabel = ylabel;
end

% ---------------------------------------------------
function varargout = getcoord(varargin)
% Calculates cursor position in axes units using 'CurrentPoint' of
% CurrentAxes. if nargout=3, return handle of this axes
app = varargin{1};
hand = [];
if nargin == 2, hand = varargin{2}; end
% Determination of positon of the pointer
CoordXY = get(app.MainFigure, 'CurrentPoint');
RePos = get(app.haReal, 'Position');
ImPos = get(app.haImag, 'Position');
ReX = 0;
ReY = 0;
ImX = 0;
ImY = 0;
if isempty(hand)
  switch app.DataPart
    case {2}
      if CoordXY(1)>RePos(1)&& CoordXY(1)< sum(RePos([1, 3])), ReX = 1; end
      if CoordXY(2)>RePos(2)&& CoordXY(2)< sum(RePos([2, 4])), ReY = 1; end
      if CoordXY(1)>ImPos(1)&& CoordXY(1)< sum(ImPos([1, 3])), ImX = 1; end
      if CoordXY(2)>ImPos(2)&& CoordXY(2)< sum(ImPos([2, 4])), ImY = 1; end
    case {1,3}
      if CoordXY(1)>RePos(1)&& CoordXY(1)< sum(RePos([1, 3])), ReX = 1; end
      if CoordXY(2)>RePos(2)&& CoordXY(2)< sum(RePos([2, 4])), ReY = 1; end
  end
  if isempty(hand)
    if ReX&&ReY, hand = app.haReal; end
    if ImX&&ImY, hand = app.haImag; end
  end
end
x=[]; y=[];
if ~isempty(hand)
  CXY = get(hand, 'CurrentPoint');
  x = CXY(1, 1);
  y = CXY(1, 2);
end
varargout{1} = x;
varargout{2} = y;
if nargout == 3
  varargout{3} = hand;
end

% ---------------------------------------------------
function [xx, yy] = GetOnScreen1DSlice(app)
if app.PlotType ~= 1
  xx = []; yy = [];
else
  switch app.Projection
    case 1 % X
      yy    = app.src.y(:,min(app.Selection,end));
      xx    = app.src.ax.x;
    case 2 % Y
      yy    = app.src.y(min(app.Selection,end), :)';
      xx    = app.src.ax.y;
  end
end

% ---------------------------------------------------
function rs = GetSlice(src, dim, sel)
src.ax.y = safeget(src.ax,'y',0);
rs = src;
switch dim
  case 1 % X
    rs.y    = src.y(:,min(sel,end));
    rs.ax.y = src.ax.y(min(sel,end));
  case 2 % Y
    rs.y    = src.y(sel,:)';
    rs.ax.x = src.ax.y;
    rs.ax.y = src.ax.x(min(sel,end));
    rs.ax.xlabel = safeget(src.ax,'ylabel', '?,');
    rs.ax.ylabel = safeget(src.ax,'xlabel', '?,');
  otherwise
end

% ---------------------------------------------------
function rs = GetSourceSlice(app)
rs = GetSlice(app.src, app.Projection, app.Selection);

% ---------------------------------------------------
function UpdateSlider(app)
n = 1; % number of points
switch app.DataSource
  case 1
    n = size(app.src.y, app.AntiProjection);
  case 2
    for jj=1:length(app.out)
      n1 = size(app.out{jj}.y, app.AntiProjection);
      if n1>n, n=n1; end
    end
  otherwise
    n = size(app.src.y, app.AntiProjection);
    for jj=1:length(app.out)
      n1 = size(app.out{jj}.y, app.AntiProjection);
      if n1>n, n=n1; end
    end
end
if n > 1
  nold = get(app.sl2Dplot, 'Max');
  if nold ~= n
    set(app.sl2Dplot, 'Min', 1, 'Max', n, 'Value', 1, 'SliderStep', [1,1]/(n-1), 'Enable', 'on');
  else
    set(app.sl2Dplot, 'Enable', 'on');
  end
else
  set(app.sl2Dplot, 'Min', 1, 'Max', 2, 'Value', 1, 'SliderStep', [1,1], 'Enable', 'off');
end
sl2Dplot_Callback(app.sl2Dplot, [], app);

% ---------------------------------------------------
function [x,y, button]=GetPoint(app, n)
set(0,'CurrentFigure', app.MainFigure)
[x,y, button]=ginput(n);

% ---------------------------------------------------
function [x,y]=GetBox(app)
set(0,'CurrentFigure', app.MainFigure)
waitforbuttonpress;
[x1, y1, hand] = getcoord(app);
rbbox;
[x2, y2] = getcoord(app,hand);
x = [x1; x2];
y = [y1; y2];

% ---------------------------------------------------
function fMoveDataBr_Callback(h, app)
app.changePart = 1;
guidata(h, app);

% ---------------------------------------------------
function src = VerifyData(src)

if isfield(src, 'y')
  if find(size(src.y) ~= 1) == 1
    src.y = src.y(:);
  end
end

src.ax = safeget(src, 'ax', []);
if isfield(src.ax, 'x')
  src.ax.x = src.ax.x(:);
end

if isfield(src.ax, 'y')
  src.ax.y = src.ax.y(:);
end

%----------------------------------------------------------------------
function app = MakeFavorites(app, mode)
switch mode
  case 'update'
    callback = 'kazan(''OnFavoriteSelected'',gcbo,[],guidata(gcbo))';
    delete(app.mFavoriteDirectoriesMain.Children);
    
    for ii=1:length(app.FavoriteDirectories)
      if ~isempty(app.FavoriteDirectories{ii})
        uimenu(app.mFavoriteDirectoriesMain, 'Label', app.FavoriteDirectories{ii}, 'Callback', callback);
      end
      app.ini.KazanViewer.(['Favorite',num2str(ii)]) = app.FavoriteDirectories{ii};
    end
    uimenu(app.mFavoriteDirectoriesMain, 'Label', 'Add Current', 'Separator', 1, 'Callback', callback);
    uimenu(app.mFavoriteDirectoriesMain, 'Label', 'Remove Current', 'Callback', callback);
    uimenu(app.mFavoriteDirectoriesMain, 'Label', 'Remove All', 'Callback', callback);
    
  case 'addcurrent'
    the_dir = app.ePath.String;
    idx = find(strcmp(app.FavoriteDirectories, the_dir));
    if ~isempty(idx)
      app.FavoriteDirectories(idx) = [];
      app.FavoriteDirectories{5} = '';
      for ii=4:-1:1, app.FavoriteDirectories{ii+1}=app.FavoriteDirectories{ii}; end
    elseif idx == 1, return;
    else
      for ii=4:-1:1, app.FavoriteDirectories{ii+1}=app.FavoriteDirectories{ii}; end
    end
    app.FavoriteDirectories{1}=the_dir;
    app = MakeFavorites(app, 'update');
  case 'removecurrent'
    the_dir = app.ePath.String;
    idx = find(strcmp(app.FavoriteDirectories, the_dir));
    if ~isempty(idx)
      app.FavoriteDirectories(idx) = [];
      app.FavoriteDirectories{5} = '';
      app = MakeFavorites(app, 'update');
    end
  case 'removeall'
    for ii=1:5, app.FavoriteDirectories{ii} = ''; end
    app = MakeFavorites(app, 'update');
end
guidata(app.MainFigure, app);

%----------------------------------------------------------------------
function app = MakePlugins(app, mode, arg1)
switch mode
  case 'close'
    Plugins = {};
    for ii=1:length(app.Plugins)
      if app.Plugins{ii} ~= arg1, Plugins{end+1}=app.Plugins{ii}; end
    end
    app.Plugins = Plugins;
    guidata(app.MainFigure, app);
    fprintf('Plugin removed. %i left.\n', length(app.Plugins))
  case 'closeall'
    for ii=1:length(app.Plugins), try delete(app.Plugins{ii}); catch, end; end
  case 'onload'
    for ii=1:length(app.Plugins)
      try
        [~,name] = fileparts(get(app.Plugins{ii}, 'FileName'));
        str = [name, '(''FileUpdate'',app.Plugins{ii},guidata(app.Plugins{ii}));'];
        try eval(str); catch err, ShowError(err); end
      catch err
        DeletePlugins(app.Plugins{ii}, app.MainFigure);
        ShowError(err, sprintf('FileUpdate error ''%s''\n', name));
      end
    end
  case 'search'
    plugins = dir(fullfile(app.ini.KazanViewer.plugin_path, '*.fig'));
    plugins2 = dir(fullfile(app.ini.KazanViewer.plugin_path, '*.mlapp'));
    [sorted_plugin_names] = sortrows({plugins.name,plugins2.name}');
    sorted_plugin_names(strcmpi(sorted_plugin_names,'kazan.fig')) = [];
    callback = 'kazan(''OnPluginSelected'',gcbo,[],guidata(gcbo))';
    
    delete(app.mPluginsMain.Children);
    
    for ii=1:size(sorted_plugin_names)
      [~,plugin_name_raw]=fileparts(sorted_plugin_names{ii});
      plugin_name = upper(plugin_name_raw);
      plugin_ini = safeget(app.ini, plugin_name, []);
      plugin_ini.favorite = safeget(plugin_ini, 'favorite', '0');
      app.ini.(plugin_name) = plugin_ini;
      favorite = str2double(plugin_ini.favorite);
      
      if favorite
        hmenu = uimenu(app.mPluginsMain, 'Label',plugin_name, 'callback', callback);
        hmenu.UserData = plugin_name_raw;
      end
    end
    
    hmenuOther = uimenu(app.mPluginsMain, 'Label', 'Other plugins');
    hmenuAddFavorites = uimenu(app.mPluginsMain, 'Label', 'Add to Favorites', 'Separator', 1);
    hmenuRemoveFavorites = uimenu(app.mPluginsMain, 'Label', 'Remove from Favorites');
    for ii=1:size(sorted_plugin_names)
      [~,plugin_name_raw]=fileparts(sorted_plugin_names{ii});
      plugin_name = upper(plugin_name_raw);
      plugin_ini.favorite = safeget(app.ini.(plugin_name), 'favorite', '0');
      favorite = str2double(plugin_ini.favorite);
      
      if ~favorite
        hmenu = uimenu(hmenuOther, 'Label',plugin_name, 'Callback', callback);
        hmenu.UserData = plugin_name_raw;
        hmenu = uimenu(hmenuAddFavorites, 'Label',plugin_name, 'Callback', callback);
        hmenu.UserData = plugin_name_raw;
      else
        hmenu = uimenu(hmenuRemoveFavorites, 'Label',plugin_name, 'Callback', callback);
        hmenu.UserData = plugin_name_raw;
      end
    end
    
    app.mPluginsArrange = uimenu(app.mPluginsMain, 'Label', 'Arrange plugins', 'Tag', 'mPluginsArrange',...
      'Separator', 1, 'Callback', callback);
    app.mPluginsReload = uimenu(app.mPluginsMain, 'Label', 'Reload plugins', 'Tag', 'mPluginsReload',...
      'Callback', callback);
end
guidata(app.MainFigure, app);

% --------------------------------------------------------------------
function DeletePlugins(phandle, MainFigure)
MakePlugins(guidata(MainFigure), 'close', phandle);

% everything related to parameters
%----------------------------------------------------------------------
function [app, data] = MakeParameters(app, mode, data)

switch mode
  case 'load' % load parameters from data structure ---------------------------------------------------
    fields = fieldnames(data);
    
    for k=1:length(fields)
      fname = fields{k};
      idx = find(strcmp( cellfun( @(a) a.title, app.View.Par, 'UniformOutput', false ), {fname}), 1);
      if ~isempty(idx)
        switch app.View.Par{idx}.type
          case 's',  app.View.Par{idx}.value = data.(fname);
          otherwise,  app.View.Par{idx}.value = num2str(data.(fname));
        end
      end
    end
    
    for k=1:length(fields)
      fname = fields{k};
      idx = find(strcmp( cellfun( @(a) a.title, app.DataParameters, 'UniformOutput', false ), {fname}), 1);
      if ~isempty(idx)
        switch app.DataParameters{idx}.type
          case 's',  app.DataParameters{idx}.value = data.(fname);
          otherwise,  app.DataParameters{idx}.value = num2str(data.(fname));
        end
      end
    end
    
    %     MakeParameters(app, 'update_list');
    MakeParameters(app, 'update_editor');
    
  case 'update_list' %-----------------------------------------------------------------------------------
    pars = [app.View.Par, app.DataParameters];
    titles = cellfun(@(a) [a.prefix,a.title], pars, 'UniformOutput', false);
    app.ddSelParameter.String = titles;
    
  case 'update_editor' %-----------------------------------------------------------------------------------
    pars = [app.View.Par, app.DataParameters];
    idx = app.ddSelParameter.Value;
    switch pars{idx}.type
      case 's',  app.eParameter.String = pars{idx}.value;
      otherwise,  app.eParameter.String = num2str(pars{idx}.value);
    end
    
  case 'from_editor' %-----------------------------------------------------------------------------------
    idx = app.ddSelParameter.Value;
    if idx <= length(app.View.Par)
      switch app.View.Par{idx}.type
        case 's',  app.View.Par{idx}.value = app.eParameter.String;
        otherwise,  app.View.Par{idx}.value = str2double(app.eParameter.String);
      end
    else
      idx = idx - length(app.View.Par);
      switch app.DataParameters{idx}.type
        case 's',  app.DataParameters{idx}.value = app.eParameter.String;
        otherwise,  app.DataParameters{idx}.value = str2double(app.eParameter.String);
      end
    end
    
    pars = [app.View.Par, app.DataParameters];
    for kk=1:length(pars)
      % fixing loaded parameters
      switch pars{kk}.type
        case 's', if ~ischar(pars{kk}.value), pars{kk}.value = num2str(pars{kk}.value); end
        case 'f', if ischar(pars{kk}.value), pars{kk}.value = str2double(pars{kk}.value); end
      end
      data.(pars{kk}.title) = pars{kk}.value;
    end
end

%----------------------------------------------------------------------
% Verify data structure, create necessary fields if missing
function Data = Verify(~, Data)
if isfield(Data, 'y')
  if find(size(Data.y) ~= 1) == 1
    Data.y = Data.y(:);
  end
end

Data.ax = safeget(Data, 'ax', []);
if isfield(Data.ax, 'x')
  Data.ax.x = Data.ax.x(:);
end

if isfield(Data.ax, 'y')
  Data.ax.y = Data.ax.y(:);
end

Data.filename = safeget(Data, 'filename', '');
sz = size(Data.y);

Data.Script = safeget(Data, 'Script', '');

if ~isfield(Data.ax, 'x'), Data.ax.x = (1:sz(1))'; end
Data.ax.xlabel = safeget(Data.ax, 'xlabel', 'unknown, pnt');
Data.ax.xunit = safeget(Data.ax, 'xunit', '');
if ~isfield(Data.ax, 'y'), Data.ax.y = (1:sz(2))'; end
Data.ax.ylabel = safeget(Data.ax, 'ylabel', 'unknown, pnt');
Data.ax.yunit = safeget(Data.ax, 'yunit', '');

Data.ax.filt = safeget(Data.ax, 'filt', 'none');
Data.ax.diff = safeget(Data.ax, 'diff', 'normal');
Data.ax.filtpar1 = safeget(Data.ax, 'filtpar1', 'auto');

%----------------------------------------------------------------------
function results = GetData3(app, mode)
switch mode
  case 'source', results = app.src;
  case 'psource', results = Process(app, app.src);
  case 'output', results = app.out;
end

%----------------------------------------------------------------------
function SetData3(app, mode, arg)
switch mode
  case 'source', app.src = arg;
    app.SrcControlUpdate();
  case 'output', app.out = arg;
    app.UpdateSlider();
end
app.ProcessMenuChanged([]);
app.View.DataSrcOut = 2;
app.bSource.Value = 0;
app.bOutput.Value = 1;
PlotData(app);

%----------------------------------------------------------------------
function results = Get(app, mode)
switch mode
  case 'font', results = app.View.Font;
  case 'slice', results = min(app.View.Position, size(app.src.y,2));
  otherwise, results = [];
end

%----------------------------------------------------------------------
function results = Set(app, mode, value)
switch mode
  case 'font', app.View.Font=value;
  case 'slice'
    app.View.Position=value;
    app.sl2Dplot.Value = value;
    PlotData(app);
  case 'cursor'
    title(app.haReal, value);
  case 'message'
    app.MessageBlocker = tic;
    title(app.haReal, value);
    guidata(app.MainFigure, app);
  otherwise, results = [];
end

%----------------------------------------------------------------------
function WindowsMenuSelected(h, ~, app, varargin)
show_idx = [ app.mToolsShowinFigure, app.mAxis1Of12, app.mAxis2Of12, app.mAxis1Of21,app.mAxis2Of21 ];

switch h
  case app.mFileNewWindow, kazan; return;
  case app.mOpenInFigure
    app.AssociatedFigure = figure;
    set(app.AssociatedFigure,'UserData', struct('MainFigure',app.MainFigure));
    pMenu = uimenu('Parent', app.AssociatedFigure, 'Label', 'KV_Tools');
    uimenu('Parent', pMenu, 'Label', 'Export Image', 'Callback', 'kv_exportimage(gcf, ''gui'')');  % alsi 12.01.05
    uimenu('Parent', pMenu, 'Label', 'Profile Manager', 'Callback', 'kv_printprofile');  % alsi 12.01.05
    uimenu('Parent', pMenu, 'Label', 'Associate',...
      'Callback', ['ud = get(gcf ,''UserData''); hand = getfield(ud,''MainFigure''); gdata = guidata(hand);',...
      'eval(''kazan(''''AssociateFigure'''',hand, gcf)'');']);
    guidata(app.MainFigure, app);
    
    %         PaperSize = get(a,'PaperSize');
    %         brd = 0.6;
    %     set(a,'PaperPosition', [brd, brd, PaperSize(1)-2*brd, PaperSize(2)-2*brd]);
  case num2cell(show_idx)
    idx = find(show_idx == h);
    mplot_n = [1, 1, 2, 1, 2];
    mvert_dim = [1,1,1,2,2];
    mhorz_dim = [1,2,2,1,1];
    
    plot_n = mplot_n(idx);
    vert_dim = mvert_dim(idx);
    horz_dim = mhorz_dim(idx);
    
    % Prepare a figure/axis for the plotting
    if isempty(app.AssociatedFigure) || h == app.mToolsShowinFigure || ~ishandle(app.AssociatedFigure)
      WindowsMenuSelected(app.mOpenInFigure, [], app, varargin);
      app = guidata(app.MainFigure);
    end
    set(0, 'CurrentFigure', app.AssociatedFigure);
    
    switch app.DataPart
      case {1, 3}
        if vert_dim*horz_dim == 1
          cla;
        else
          subplot(vert_dim, horz_dim, plot_n)
          cla;
        end
        plot(1);
        imh = gca;
      case 2
        clf;
        subplot(2,1,2)
        plot(1);
        imh = gca;
        subplot(2,1,1)
        plot(1);
    end
    
    %   Plotting
    PlotData(h, gca, imh, app.AssociatedFigure);
    
    %   workaround for fix scale problem
    axis tight
    axis(imh,'tight')
    
    % Adding labels
    label = get(app.tUnits, 'String');
    title(app.fname, 'Interpreter', 'none')
    ylabel('real');
    xlabel(label);
    
    if app.DataPart==2
      subplot(2,1,2);
      ylabel('imag');
      xlabel(label);
    end
  case app.m2DSurf
    figure; hold on;
    [X,Y] = meshgrid(app.src.ax.y,app.src.ax.x);
    s = surf(X,Y,app.src.y);
    s.EdgeColor = 'none';
    xlabel(app.src.ax.xlabel)
    ylabel(app.src.ax.ylabel)
    colormap jet;
    view(30,14)
  case app.m2DScatter
    figure; hold on;
    for ii=1:size(app.src.y,2)
      x = app.src.ax.x;
      y = app.src.y(:,ii);
      z = app.src.ax.y(ii)*ones(size(x));
      scatter3(x,z,y,0.25,y,'filled');
    end
    xlabel(app.src.ax.xlabel)
    ylabel(app.src.ax.ylabel)
    colormap jet;
    view(-55,14)
  case app.m2DSurface
    figure; hold on;
    for ii=1:size(app.src.y,2)
      x = app.src.ax.x';
      y = app.src.y(:,ii)';
      z = app.src.ax.y(ii)*ones(size(x));
      surface([x;x],[z;z],[y;y],[y;y],...
        'FaceColor','none',...
        'EdgeColor','interp');
    end
    xlabel(app.src.ax.xlabel)
    ylabel(app.src.ax.ylabel)
    colormap jet;
    view(-55,14)
    
  case app.mToolsAssociatedWindowReset
    AssociateFigure(app.MainFigure,[]);
    
  case app.mToolsSkyline
    h = figure;
    xlim = get(app.haReal, 'XLim');
    ylim = get(app.haReal, 'YLim');
    app.src.ax.Lim = [xlim, ylim] ;
    switch app.DataPart
      case 1
        % Parameter Lim need to be traced-> wrong set
        app.src.ax.Lim = [min(app.src.ax.x), max(app.src.ax.x), min(app.src.ax.y), max(app.src.ax.y)];
        kvskyline(h, app.src.ax, app.src.y, 'plot', 'image');
      case 3
        % Parameter Lim need to be traced-> wrong set
        app.src.ax.Lim = [min(app.src.ax.x), max(app.src.ax.x), min(app.src.ax.y), max(app.src.ax.y)];
        kvskyline(h, app.src.ax, abs(app.src.y), 'plot', 'image');
    end
    uimenu('Parent', h, 'Label', 'Export Image (KV)', 'Callback', 'kv_exportimage(gcf, ''gui'')');  % alsi 12.01.05
  case app.mToolsSkyline1
    h = figure;
    xlim = get(app.haReal, 'XLim');
    ylim = get(app.haReal, 'YLim');
    for ii = GetPlotNumber(app)
      if ii~=0
        % output
        app.out{ii}.ax.Lim = [xlim, ylim] ;
        newax = getplotaxis(app.xrepresent, app.yrepresent, app.out{ii}.ax);
        switch app.DataPart
          case {1,2}
            kvskyline(h, newax, app.out{ii}.y, 'plot', 'contour', 'hold', 'on');
          case 3
            kvskyline(h, newax, abs(app.out{ii}.y), 'plot', 'contour', 'hold', 'on');
        end
      else
        % source
        app.src.ax.Lim = [xlim, ylim] ;
        newax = getplotaxis(app.xrepresent, app.yrepresent, app.src.ax);
        switch app.DataPart
          case {1,2}
            app.src.ax.Lim = [min(app.src.ax.x), max(app.src.ax.x), min(app.src.ax.y), max(app.src.ax.y)];
            kvskyline(h, newax, app.src.y, 'plot', 'contour');
          case 3
            kvskyline(h, newax, abs(app.src.y), 'plot', 'contour');
        end
      end
    end
    uimenu('Parent', h, 'Label', 'Export Image (KV)', 'Callback', 'kv_exportimage(gcf, ''gui'')');  % alsi 12.01.05
end

% ---------------------------------------------------
function AssociateFigure(h, a_fig)
hand = guidata(h);
hand.AssociatedFigure = a_fig;
if isempty(a_fig)
  set(hand.mToolsAssociatedwindow, 'Label', 'No Figure Is Associated');
else
  set(hand.mToolsAssociatedwindow, 'Label', ['Associated Figure -> ',num2str(gcf)]);
end
guidata(h,hand);

