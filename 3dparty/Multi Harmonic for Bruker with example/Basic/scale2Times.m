function new=scale2Times(data2D,cut)

ss=size(data2D); s1=ss(1); s2=ss(2);
new=zeros( round(s1/2) ,s2);

for i=1:round(s1/2);
   if 2*i<=s1
   new(i,:)=(    data2D(2*i-1,:)+data2D(2*i,:)   )/2;
   else
   new(i,:)=data2D(2*i-1,:);
   end;
        
end;


