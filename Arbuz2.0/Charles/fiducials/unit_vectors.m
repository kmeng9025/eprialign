function unitvecs=unit_vectors(endpoints)
%
% find unit vectors along directions connecting pairs of points
% input data is N rows of 3 columns
% output is N/2 rows of 3 columns
% unit vectors point from row 1 to row 2, row 3 to row 4, etc.
% in the list of input vectors.
%

nvecs=size(endpoints,1)/2;
unitvecs=zeros(nvecs,3);
x=reshape(endpoints(:,1),2,nvecs);
y=reshape(endpoints(:,2),2,nvecs);
z=reshape(endpoints(:,3),2,nvecs);

for i=1:nvecs
    vec=[x(2,i)-x(1,i) y(2,i)-y(1,i) z(2,i)-z(1,i)];
    vlen=sqrt(dot(vec,vec));
    unitvecs(i,:)=vec/vlen;
end