function [a1,a2,a3,center,svals]=principal_axis(pointlist)

center=sum(pointlist,1)/size(pointlist,1);

x=pointlist(:,1)-center(1);
y=pointlist(:,2)-center(2);
z=pointlist(:,3)-center(3);

m=zeros(3,3);
m(1,1)=dot(x,x);
m(1,2)=dot(x,y);
m(1,3)=dot(x,z);
m(2,1)=dot(y,x);
m(2,2)=dot(y,y);
m(2,3)=dot(y,z);
m(3,1)=dot(z,x);
m(3,2)=dot(z,y);
m(3,3)=dot(z,z);

[u,s,v]=svd(m);
a1=u(:,1);
a2=u(:,2);
a3=u(:,3);
svals = [s(1,1) s(2,2) s(3,3)];