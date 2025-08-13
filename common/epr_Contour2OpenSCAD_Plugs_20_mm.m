function epr_Contour2OpenSCAD_Plugs_20_mm(mod_name, fname, vectors, scale, UVShift, inverted_plug_wall)

Plug_height = 6;

inverted = false;
if inverted_plug_wall > 0
  inverted = true;
end

endl = sprintf('\n');
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

%  //constants 
%  Plug_height = 10; 
%  Plug_OD = 18.0; 
%  Notch_parameter = 2.55; 
%  
%  //plug with arb hole for hypoxia targeting 
%  module Plug_model(){ 
%  cylinder(d = Plug_OD,h = Plug_height); 
% 
%  translate([-Plug_OD/2-1 ,-Notch_parameter/2,2]) 
%   cube([Notch_parameter, Notch_parameter, 4],true);
% 
% }

str = [str,' //constants ',endl];
str = [str,' Plug_height = ',num2str(Plug_height),'; ',endl];
str = [str,' Plug_support = 0.5; ',endl];
%Changed plug height 5/2/16 -MM
%str = [str,' Plug_OD = 15; ',endl];
str = [str,' Plug_OD = 20.1; ',endl];
str = [str,' Notch_parameter = 4.6; ',endl];
str = [str,' ',endl];

if ~inverted
  str = [str,' //plug with arb hole for hypoxia targeting ',endl];
  str = [str,' module Plug_model(){ ',endl];
  str = [str,' cylinder(d = Plug_OD,h = Plug_height); ',endl];
  str = [str,' translate([-Plug_OD/2-1.25 ,0,2]) ',endl];
  str = [str,'  cube([Notch_parameter-0.5, Notch_parameter, 4],true); } ',endl];
  str = [str,' ',endl];
  
  
  str = [str,' ',endl];
  str = [str,' ',endl];
  str = [str,' difference(){ ',endl];
  str = [str,' Plug_model();',endl];
  str = [str,' translate([0,0,-1]) ',endl];
  str = [str,' Plug_cut(20); ',endl];
  str = [str,'}',endl];
else
  str = [str,...
    ' //plug with arb hole for hypoxia targeting ',endl,...
    ' module Plug_model(){ ',endl,...
    '   translate([-Plug_OD/2-1.25,0,2]) ',endl,...
    '     cube([Notch_parameter-0.5, Notch_parameter, 4],true);',endl,...
    ' }',endl,...
    '',endl,...
    ' Plug_model();',endl,...
    ' Plug_cut(Plug_height); ',endl,...
    ' ',endl,...
    ' difference() {',endl,...
    '  cylinder(d = Plug_OD,h = Plug_height); ',endl,...
    '  cylinder(d = Plug_OD - ',num2str(inverted_plug_wall),',h = Plug_height); ',endl,...
    '  } ',endl,...
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
