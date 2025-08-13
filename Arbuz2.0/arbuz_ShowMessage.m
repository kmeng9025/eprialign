% ARBUZ_SHOWMESSAGE  display message in the standart dispaly areas
% arbuz_ShowMessage(hGUI, the_message);
% hGUI - handle to the object that holds the project [double]
% the_message - message [string]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function arbuz_ShowMessage(hGUI, the_message)

if hGUI ~= 0
  handles = guidata(hGUI);
  if isempty(handles), return; end
  if ~isfield(handles, 'eLogWindow'), return; end
  
  str = get(handles.eLogWindow, 'string');
  if ~iscell(str), str = {str}; end
  str{end+1} = the_message;
  if length(str) > 5
    str = str(length(str)-(5-1):end);
  end
  
  set(handles.eLogWindow, 'string', str);
end
disp(the_message);
drawnow;
