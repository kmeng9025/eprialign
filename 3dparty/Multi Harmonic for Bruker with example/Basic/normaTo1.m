function new=normaTo1(data2D)

ss=size(data2D); s1=ss(1); s2=ss(2);

new=zeros(ss);

for i=1:s1
       tmp=sum( data2D(i,:) );
       new(i,:)= data2D(i,:)/tmp*1;
end;


