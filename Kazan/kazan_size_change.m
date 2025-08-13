function kazan_size_change(hGUI, font_size)

allobj = findobj(hGUI, 'Style', 'pushbutton');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'togglebutton');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'edit');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'popupmenu');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'list');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'menu');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'text');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');
allobj = findobj(hGUI, 'Style', 'checkbox');
set(allobj, 'FontSize', font_size, 'FontUnits', 'points');