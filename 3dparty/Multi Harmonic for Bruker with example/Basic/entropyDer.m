function res=entropy(Image);
inx2=isnan(Image);
Image(inx2)=0;

inx1=(Image*1000<=0);
S=sum(sum(Image));

% Log(0)=- infinity
Image=Image/S;
Image(inx1)=1;

%%%%%%%%%%%%%%%%%%%%


A=log(Image); 
B=entropy(Image);
C=A+B;
res=-C/S;
