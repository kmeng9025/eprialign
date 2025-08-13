function res1D=reSweep(data1D,newSweep,oldSweep)

%function res1D=reSweep(data1D,newSweep,oldSweep)
% эмулирует измнение sweep_а на спектрометре.
% сохран** кол=во точек

L2=length(data1D); % Length
x=data1D;


ratioSweep=newSweep/oldSweep;

if newSweep<=oldSweep
 newNumberOfPoints=L2*ratioSweep;
 cutNumber=round( (L2-newNumberOfPoints)/2 ); cn=cutNumber;

 xxx=x( (cn+1):(L2-cn)  );
 %  plot(x);
 Lxxx=length(xxx);
 yyy=interp1(1:Lxxx,xxx,1:(Lxxx-1)/(L2-1):Lxxx);
 % plot(yyy);
 res1D=yyy';
else
 newNumberOfPoints=L2*ratioSweep;
 addNumber=round( (newNumberOfPoints-L2)/2 ); cn=addNumber; 
 z=zeros(1,cn);
 xxx=[z x' z];xxx=xxx';
 %  plot(x);
 Lxxx=length(xxx);
 yyy=interp1(1:Lxxx,xxx,1:(Lxxx-1)/(L2-1):Lxxx);
 % plot(yyy);
 res1D=yyy';
end