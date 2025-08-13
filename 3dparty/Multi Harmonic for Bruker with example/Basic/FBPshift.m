function res=FBPshift(FBP)   
s=size(FBP);
Nimage=s(1);
 for i=1:Nimage
        tmp=FBP(:,i);
        FBP(:,i)=rotatev(tmp',1)';
 end
    
 res=FBP;