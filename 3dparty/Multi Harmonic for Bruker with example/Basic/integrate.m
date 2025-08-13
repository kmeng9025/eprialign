function data2D=shiftData %loadDatAndMetki
clear
load data;
load par;

imagesc(data2D);

 %выравнивание по меткам
clear shift;
% sgn=ones(1,29); for i=1:14 sgn(i)=-1; end;
% grad=grad.*sgn;

ss=size(data2D); s1=ss(1); s2=ss(2);
shift=round( (sprava + sleva-dHPoints)/2- s1/2 );  %%%!!!!   300
for i=1:32
   tmp=data2D(:,i); 
   tmp1=shift(i);
   data2D(:,i)=rotatev(tmp',tmp1)';    
end;

% % предварительное изменение sweep_а дл* последующего выравнивани*
% 
% for i=1:29
%    tmp=data2D(:,i); 
%    newSweep=sw1(i);
%    data2D(:,i)=reSweep(tmp,newSweep,sw);    
% end;

% общий сдвиг 
% for i=1:29
%    tmp=data2D(:,i); 
%    data2D(:,i)=rotatev(tmp',-300)';    
% end;

% function res1D=reSweep(data1D,newSweep,oldSweep)
% эмулирует измнение sweep_а на спектрометре.
% сохран** кол=во точек
% save data data2D sleva sprava grad shift0

imagesc(data2D);
% plot(data2D(:,15))