%totalL=50;

num=str2num( get(t1,'String'));
step=str2num( get(t2,'String'));

% gradient=sprava -sleva;

slid=get(s1,'Value');
ss=size(newP);
Newslid=round(ss(1)*slid)



% shfta=round((get(s1,'Value')-0.5)*sweep);
% shftb=round((get(s2,'Value')-0.5)*sweep);
% 
% a=projs(:,num)';
% b=projs(:,totalL+1-num)';
% 
% shA=rotatev(a,shfta+shftb);
% shB=fliplr(rotatev(b,shfta-shftb));
% new=integ(real(fromImToComplex((shA+shB))));
% 
% axes(axe1); plot(shA);
% axes(axe2); plot(shB);
% axes(axe3); plot(new);

