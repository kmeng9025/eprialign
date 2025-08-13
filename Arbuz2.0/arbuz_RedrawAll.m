% ARBUZ_REDRAWALL draw all the viewes registered
% arbuz_RedrawAll(hGUI);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function arbuz_RedrawAll(hGUI)

hhandles = guidata(hGUI);
FigN = fix(str2double(get(hhandles.eViewerFigure, 'string')));

% find out which scene we are showing
if isfield(hhandles, 'pmShowGroup')
  val = get(hhandles.pmShowGroup, 'Value');
  
  if val == 1
    options.images = arbuz_FindImage(hGUI, 'v', '', '', {});
    options.FigN = FigN;
    
    % find transformation that leads to the frame of interest
    if isfield(hhandles, 'pmReferenceFrame')
      val = get(hhandles.pmReferenceFrame, 'Value');
      str = get(hhandles.pmReferenceFrame, 'String');
    else
      val = 1;
    end
    
    if val == 1
      options.A2frame = eye(4);
    else
      find_origin = arbuz_FindImage(hGUI, 'all', 'FullName', str{val}, {'Ashow'});
      if isempty(find_origin), return; end
      options.A2frame = inv(find_origin{1}.Ashow);
    end
    
    arbuz_3DRDR(hGUI, options)
  elseif val == 2
    % show nothing
  else
    saves = arbuz_get(hGUI, 'saves');
    if val-2 <= length(saves)
      options = saves{val-2};
      options.FigN = FigN;
      eval([options.renderer, '(hGUI, options);']);
    end
  end
else
  options.images = arbuz_FindImage(hGUI, 'v', '', '', {});
end

% check plugins
ActiveReports = hhandles.ActiveReports(ishandle(hhandles.ActiveReports));
for ii=1:length(ActiveReports)
  [pp, FileName] = fileparts(get(ActiveReports(ii), 'FileName'));
  eval([FileName,'(''ForceRedraw'',ActiveReports(ii))'])
end
