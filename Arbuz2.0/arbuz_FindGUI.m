function hh = arbuz_FindGUI()

set(0, 'ShowHiddenHandles', 'on');
h = findobj('Tag', 'MainFigure');
set(0, 'ShowHiddenHandles', 'off');

hh = [];
for ii=1:length(h)
%   gd = guidata(h(ii));
  if strfind(h.Name, 'ArbuzGUI')
    hh(end+1) = h(ii);
  end
end