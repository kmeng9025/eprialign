n=16;

a=rand(1,1000); 
std(a)

b=rand(n,1000); 
b=sum(b);

std(b)/std(a)/sqrt(n)
