function footer(optionalstring)

h = axes('position',[0.02 0.02 eps eps]);
set(h,'visible','off');
set(h,'tag','myfooter');
text(0,0,[datestr(now,2) '  ' datestr(now,14) ': ' optionalstring],'Interpreter','none','FontSize',8)