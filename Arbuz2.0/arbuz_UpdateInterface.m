% ARBUZ_UPDATEINTERFACE  updates gui interface, reflects changes to the project
% arbuz_UpdateInterface(hGUI);
% hGUI - handle to the object that holds the project [double]

% Author: Boris Epel
% Center for EPR imaging in vivo physiology
% University of Chicago, MAY 2014
% Contact: epri.uchicago.edu

function arbuz_UpdateInterface(hGUI)

% Only list box requires update
ArbuzGUI('UpdateListbox', 0, guidata(hGUI));
