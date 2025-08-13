% ARBUZ_GET  get project property
% result = arbuz_get(hGUI, property_name);
% hGUI - handle to the object that holds the project [double]
% property_name  - project property [string]
% INTERNAL:
%    STATE / FILENAME / CANCLOSE / HIGHLIGHTED / SEQUENCES / 
%    ACTIONS / ACTIONLIST / SEQUENCE / STAGES / ACTIVESEQUENCE /
%    TRANSFORMATIONS / ACTIVETRANSFORMATION / ACTIVETRANSFORMATIONNAME /
%    WATCHTRANSFORMATION / GROUPS / COORDINATES / SAVES  / IMAGENAMES
% USER LEVEL:
%    STATUS / COMMENTS
% 
% result - property value [variant]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function  res = arbuz_get(hGUI, property_name)

% load the project
prj = getappdata(hGUI, 'project');
res = [];

switch upper(property_name)
  case 'STATE', res = safeget(prj, 'state', 1);
  case 'FILENAME', res = safeget(prj, 'FileName', '');
  case 'CANCLOSE', res = safeget(prj, 'can_close', '');
  case 'HIGHLIGHTED', res = prj.highlighted;
  case 'SEQUENCES', res = prj.Sequences;   
  case 'STATUS', res = safeget(prj, 'Status', []); 
    if isempty(res), res = containers.Map; end
  case 'ACTIONS' 
    if prj.ActiveSequence == -1
      res = {};
    else
      res = prj.Sequences{prj.ActiveSequence}.Actions;
    end
  case 'ACTIONLIST'
    if prj.ActiveSequence == -1
      res = {};
    else
      act = prj.Sequences{prj.ActiveSequence}.Actions;
      for ii=1:length(act)
        res{end+1} = act{ii}.Description;
      end
    end
  case 'SEQUENCE'
    if prj.ActiveSequence == -1
      res = [];
    else
      res = prj.Sequences{prj.ActiveSequence};
    end
  case 'STAGES'
    if prj.ActiveSequence ~= -1
      seq = prj.Sequences{prj.ActiveSequence};
      res = {};
      for ii=1:length(seq.Sequence)
        res{end+1}=['Stage ',num2str(ii),': ', seq.Sequence{ii}.Name];
      end
    end
  case 'ACTIVESEQUENCE', res = prj.ActiveSequence; 
  case 'TRANSFORMATIONS', res = prj.Transformations;
  case 'ACTIVETRANSFORMATION'
    if prj.ActiveSequence == -1
      res = -1;
    else
      seq = prj.Sequences{prj.ActiveSequence};
      res = seq.ActiveTransformation;
    end
  case 'ACTIVETRANSFORMATIONNAME'
    if prj.ActiveSequence == -1
      res = -1;
    else
      seq = prj.Sequences{prj.ActiveSequence};
      if seq.ActiveTransformation == -1
        seq.ActiveTransformation = 0;
        prj.Sequences{prj.ActiveSequence}.ActiveTransformation = 0;
      end
      res = seq.Sequence{seq.ActiveTransformation}.Name;
    end
  case 'WATCHTRANSFORMATION'
    if prj.ActiveSequence >= 1 && prj.ActiveSequence <= length(prj.Sequences)
      res = prj.Sequences{prj.ActiveSequence}.WatchTransformation;
    else
      res = -1;
    end
  case 'GROUPS', res = prj.Groups;
  case 'COORDINATES', res = prj.Coordinates;
  case 'SAVES', res = prj.saves;
  case 'COMMENTS', res = safeget(prj, 'comments', '');
  case 'IMAGENAMES'
    res = {};
    for ii=1:length(prj.images), res{end+1} = prj.images{ii}.Name; end 
  otherwise
    arbuz_ShowMessage(hGUI, ['arbuz_get: Unknown property ''', property_name, '''.']);
end