% ARBUZ_SETACTIVETRANSFORMATION recalculates image transformations
% status = arbuz_SetActiveTransformation(hGUI, ActiveSequence, ActiveTransformation);
% status = arbuz_SetActiveTransformation(hGUI, ActiveSequence);
% status = arbuz_SetActiveTransformation(hGUI);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function status = arbuz_SetActiveTransformation(hGUI, ActiveSequence, ActiveTransformation)

% load the project
prj = getappdata(hGUI, 'project');
status = 1;

if nargin == 1
  ActiveSequence = prj.ActiveSequence;
  if ActiveSequence == -1, return; end
end

if nargin == 3
  if ischar(ActiveTransformation)
    for ii=1:length(prj.Sequences{prj.ActiveSequence}.Sequence)
      if strcmp(prj.Sequences{prj.ActiveSequence}.Sequence{ii}.Name, ActiveTransformation)
        prj.Sequences{prj.ActiveSequence}.ActiveTransformation = ii; break;
      end
    end
  else
    prj.Sequences{ActiveSequence}.ActiveTransformation = ActiveTransformation;
  end
end
setappdata(hGUI, 'project', prj);

seq = prj.Sequences{ActiveSequence};

% reset all transformations
arbuz_ApplyTransformation(hGUI, '', 'reset');

if seq.ActiveTransformation > 0
  % set transformations before the active
  %   hhandles = arbuz_UpdateLinkedTransformations(hGUI);
  ii = 1;
  while ii <= length(seq.Sequence) && ii < seq.ActiveTransformation
    arbuz_ApplyTransformation(hGUI, seq.Sequence{ii}.Name, 'previous');
    ii = ii + 1;
  end
  
  % set active transformation
  arbuz_ApplyTransformation(hGUI, seq.Sequence{ii}.Name, 'A');
  ii = ii + 1;
  
  while ii <= length(seq.Sequence) && ii <= seq.WatchTransformation
    arbuz_ApplyTransformation(hGUI, seq.Sequence{ii}.Name, 'next');
    ii = ii + 1;
  end
end

