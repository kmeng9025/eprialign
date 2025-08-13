function [sleva,sprava,doliaL,doliaR,IndexR,IndexL]=findNeighbours(values,number);
% a=values;
% b=number;

a=values;
b=number;

c=a(a>b); cLe=length(c);
d=a(a<b); dLe=length(d);
x=a(a==b);

    
sprava= min(c); % next to the right
IndexR=dLe+1;

sleva= max(d); % next to the right
IndexL=dLe;

% [sleva,IL] = max(a(a<b)); % netx to the left
% ILx=IL

if sleva==[] 
    sleva=0;
end
if sprava==[] 
    sprava=0;
end
if IndexR>length(a)
   IndexR=length(a);
end
    
doliaR=(-sleva+number)/(sprava-sleva); % v prozentax 
doliaL=(sprava-number)/(sprava-sleva);

if number>max(a)
    doliaR=0;
    doliaL=1;
end

if number<min(a)
    doliaR=1;
    doliaL=0;
end

if x~=[]
    IndexL=dLe+1;
    IndexR=IndexL;
    doliaL=1;
    doliaR=0;
end
 i=IndexR*doliaR+IndexL*doliaL

