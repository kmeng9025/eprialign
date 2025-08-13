function res=dif1D(data1D) %loadDatAndMetki

ss=length(data1D);

   tmp=data1D; 
   tmp1=tmp-rotatev(tmp,1);
   tmp1(1)=tmp1(2);
   tmp1(ss)=tmp1(ss-1);
   res=-tmp1;
%   res=-zeroLine(tmp1,.05);

