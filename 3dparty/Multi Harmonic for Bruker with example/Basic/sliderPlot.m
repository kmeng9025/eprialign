% slider
 ha=figure;
 s1=uicontrol(ha,'Style','slider','Position',[ss+40,ss+60,150,20],'Value',1);
 set(s1,'Callback','getvalues;');