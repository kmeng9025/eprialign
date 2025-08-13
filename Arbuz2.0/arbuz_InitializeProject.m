% ARBUZ_INITIALIZEPROJECT  prepare project data structures
% arbuz_InitializeProject(hGUI);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function arbuz_InitializeProject(hGUI)

prj = [];
prj.images = {};
prj.Sequences = {};
prj.ActiveSequence = -1;
prj.Transformations = {};
prj.ActiveTransformation = -1;
prj.Groups = {};
prj.Coordinates = {};
prj.saves = {};

prj.highlighted = [];
prj.is_locked = 0;
prj.FileName = '';
prj.can_close = true;

setappdata(hGUI, 'project', prj)