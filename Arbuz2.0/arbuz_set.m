% ARBUZ_SET  get project property
% result = arbuz_get(hGUI, property_name, property_value);
% hGUI - handle to the object that holds the project [double]
% property_name  - project property [string]
%   FILENAME
% status - 0/1 for error/success [int]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function  status = arbuz_set(hGUI, property_name, property_value)

% load the project
prj = getappdata(hGUI, 'project');
status = 1;

% set fields prior to storing to project
switch upper(property_name)
  case 'FILENAME', prj.FileName = property_value;
  case 'HIGHLIGHTED', prj.highlighted = property_value;
  case 'SEQUENCES', prj.Sequences = property_value;
  case 'ACTIONS'
    if prj.ActiveSequence >= 1 && prj.ActiveSequence <= length(prj.Sequences)
      prj.Sequences{prj.ActiveSequence}.Actions = property_value;
    end
  case 'COORDINATES', prj.Coordinates = property_value;
  case 'ACTIVESEQUENCE', prj.ActiveSequence = property_value;
  case 'TRANSFORMATIONS', prj.Transformations = property_value;
  case 'ACTIVETRANSFORMATION' % nothing here
  case 'WATCHTRANSFORMATION'
    if prj.ActiveSequence >= 1 && prj.ActiveSequence <= length(prj.Sequences)
      if ischar(property_value)
        for ii=1:length(prj.Sequences{prj.ActiveSequence}.Sequence)
          if strcmp(prj.Sequences{prj.ActiveSequence}.Sequence{ii}.Name, property_value)
            prj.Sequences{prj.ActiveSequence}.WatchTransformation = ii; break;
          end
        end
      else
        prj.Sequences{prj.ActiveSequence}.WatchTransformation = property_value;
      end
    end
  case 'GROUPS', prj.Groups = property_value;
  case 'COMMENTS', prj.comments = property_value;
  case 'STATUS'
    if contains(class(property_value), 'containers.Map')
      prj.Status = property_value;
    end
  otherwise
    arbuz_ShowMessage(hGUI, ['arbuz_set: Unknown property ''', property_name, '''.']);
    status = 0;
end

prj.can_close = 0;
setappdata(hGUI, 'project', prj);

% post setting steps
switch upper(property_name)
  case {'SEQUENCES', 'WATCHTRANSFORMATION'},
    arbuz_SetActiveTransformation(hGUI, prj.ActiveSequence, arbuz_get(hGUI, 'ActiveTransformation'));
  case 'ACTIVETRANSFORMATION',
    arbuz_SetActiveTransformation(hGUI, prj.ActiveSequence, property_value);
end