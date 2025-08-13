%общий сдвиг 
function res2D=allRotate(data2D,step) 

ss=size(data2D);
n=ss(2);

res2D=data2D;
for i=1:n
   tmp=data2D(:,i); 
   res2D(:,i)=rotatev(tmp',step)';    
end;
