function new=normaTo100(data2D)

ss=size(data2D); s1=ss(1); s2=ss(2);

new=data2D;

for i=1:s2
       tmp=sum( data2D(:,i) );
       new(:,i)= data2D(:,i)/tmp*100;
end;


