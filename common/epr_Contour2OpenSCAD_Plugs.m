function epr_Contour2OpenSCAD_Plugs(mod_name, fname, vectors, scale, UVShift, inverted_plug_wall)

%Notes and Explainer

%This function is build to solve a problem in how to connect matlab to
%openscad. Turns out Openscad can be written to like a text file and given
%the .scad extention. Below is basically a template for the plug production
%in openscad. The plug itself is a static object that has the aperature
%diffrenced from it. Each diffrent type of plug has a function that is
%cloned from this master template, they all work the same way.

%Important note: Look at around line 50 of this function to see that the
%plug diameter and hieght are NOT varible. They are static, and if the
%printer starts failing to print the correct size ( because of bad nozzle
%probably) You may need to adjust these size varibles.

%Varagin
%mod_name is just the name of the module. This is almost always Plug_cut.
%fname is the filename you want the plug written to. We format this in the
%IMRT_n_port_planning to have a very descriptive name.
%Vectors is a list of vectors that represent the 2D contours we will cut out
%of the plug. The file can accept any number of contours.
%Scale should be a static scale that relates the space of the vectors to
% "real space" 1mms. Because we use a very fine grid to produce the bev
% masks this should always be: Scale_factor = [0.025, 0.025, 1];
% UVShift is the two numbers that repesent the non-ideality of the machine
% The numbers are estimates produced by matt maggio and scott trinkle.

%Varagout = None

inverted = false;
if inverted_plug_wall > 0
  inverted = true;
end

Plug_height = 6;

endl = newline;
str = ['//',mod_name,'(10);', endl, endl];
str = [str,'module ',mod_name,'(h)', endl];
str = [str,'{',endl];
str = [str,sprintf('%s','translate([',num2str(UVShift(1)),',',num2str(UVShift(2)),',','0])'),endl] ;
str = [str, sprintf(' scale([%g, %g, %g])', scale(1), scale(2), scale(3)),endl];
str = [str, sprintf('rotate([0,0,270])',endl)] ;
str = [str,' union() {', endl];

for k = 1:length(vectors)
  
  str = [str,'linear_extrude(height=h) polygon([', endl];
  vector = vectors{k};
  
  separator = '';
  for i=1:size(vector,1)
    str = [str,separator,sprintf('[%g,%g]',vector(i,1),vector(i,2))];
    separator = ',';
  end
  str = [str,']);', endl];
end

str = [str,'}',endl];
str = [str,'}',endl];

str = [str,' ',endl]; %add lines for readablity
str = [str,' ',endl];
str = [str,' ',endl];

str = [str,' // special variables for the properties of circular objects ',endl];
str = [str,' $fa = 0.01; ',endl];
str = [str,' $fs = 0.5;',endl];

str = [str,' //constants ',endl];
str = [str,' Plug_height = ',num2str(Plug_height),'; ',endl];
str = [str,' Plug_support = 0.5; ',endl];
%Changed plug height 5/2/16 -MM
str = [str,' Plug_OD = 15.8; ',endl]; % Changed from 15.3 BE 12/16/2021
str = [str,' Notch_parameter = 2.5; ',endl];
str = [str,' ',endl];

if ~inverted
  str = [str,' //plug with arb hole for hypoxia targeting ',endl];
  str = [str,' module Plug_model(){ ',endl];
  str = [str,' cylinder(d = Plug_OD,h = Plug_height); ',endl];
%   str = [str,' translate([-((Plug_OD/2)-(Notch_parameter/1.5)),0,Plug_height]){ ',endl];BE 12/16/2021
  str = [str,' translate([-((7.95)-(Notch_parameter/1.5)),0,Plug_height]){ ',endl];
  str = [str,' translate([0,0,-Plug_height]) ',endl];
  str = [str,' linear_extrude(height = Plug_height/2, scale = 1.0) ',endl];
  str = [str,' rotate([0,0,180]) ',endl];
%   str = [str,' circle(Plug_OD/3-0.3,$fn = 3, center = true); ',endl];BE 12/16/2021
  str = [str,' circle(4.8,$fn = 3, center = true); ',endl];
  str = [str,'}',endl];
  str = [str,'}',endl];
  str = [str,' difference(){ ',endl];
  str = [str,' Plug_model();',endl];
  str = [str,' translate([0,0,-1]) ',endl];
  str = [str,' Plug_cut(20); ',endl];
  str = [str,'}',endl];
else
  str = [str,...
    ' //plug with arb hole for hypoxia targeting',endl,...
    ' module Plug_model(){',endl,...
    '   difference() {',endl,...
    '     union() {',endl,...
    '       translate([-((Plug_OD/2)-(Notch_parameter/1.5)),0,Plug_height]){',endl,...
    '         translate([0,0,-Plug_height])',endl,...
    '         linear_extrude(height = Plug_height/2, scale = 1.0)',endl,...
    '           rotate([0,0,180])',endl,...
    '             circle(Plug_OD/3-0.3,$fn = 3, center = true);',endl,...
    '         }',endl,...
    '       }',endl,...
    '   cylinder(d = Plug_OD,h = Plug_height+0.5);'...
    '   }',endl,...
    ' }',endl,...
    '',endl,...
    ' Plug_model();',endl,...
    ' Plug_cut(Plug_height);',endl,...
    '',endl,...
    '  difference() {',endl,...
    '    cylinder(d = Plug_OD,h = Plug_height);',endl,...
    '    cylinder(d = Plug_OD - ',num2str(inverted_plug_wall),',h = Plug_height);',endl,...
    '   }',endl,...
    ''];  
end

fid = fopen(fname, 'w+');

if fid == -1
  slashes = strfind(fname,'\');
  path = fname(1:slashes(end));
  status = mkdir(path);
  fid = fopen(fname, 'w+');
end

fwrite(fid, str);
fclose(fid);
