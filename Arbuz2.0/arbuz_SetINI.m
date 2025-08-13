function arbuz_SetINI(hGUI, ini)

hhandles = guidata(hGUI);
hhandles.ini = ini;
guidata(hGUI, hhandles);