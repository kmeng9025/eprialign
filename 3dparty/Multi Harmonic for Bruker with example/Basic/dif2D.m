function res=dif2D(data2D) %loadDatAndMetki

ss=size(data2D); s1=ss(1); s2=ss(2);
res=zeros(ss);
for i=1:s1
   tmp=data2D(:,i); 
   % plot(tmp);
   tmp1=tmp-rotatev(tmp',1)';
   tmp1(1)=tmp1(2);
   tmp1(s2)=tmp1(s2-1);
   res(:,i)=-zeroLine(tmp1',.05)';
%    txp=zeroLine( tnp', 0.05)';
end;

imagesc(res);
% plot(data2D(:,15))