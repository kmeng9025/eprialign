% newPrj from Sandra
function R=SandraRadon(image,theta);
%load phantom;
%addpath c:/mark/work
clc
Ph=image;
% subplot(2,1,1);
% imagesc(Ph)
ss=size(Ph);
N=ss(1); % size of Phantom
n=N;ns=N;

NPT=length(theta); % nuber of projections

dang=180/NPT;
ang0=0.5*dang;

proju=zeros(N,NPT);

sqrt2=sqrt(2);
nd2=N/2;
nless=n-1;
rl1=nd2*sqrt2;
sens=n*sqrt2/(ns-1)

% Lim1=1;% number of first non-missing prj
% Lim2=NPT;% number of last non-missing prj


for pn=1:NPT
    
% DefAngles
  % priAng=(pn-1)*dang+ang0;
  priAng=theta(pn) ;
  wway=0;
  if (priAng>89.5)&(priAng<=90);  priAng=89.5; end;
  if (priAng>90)&(priAng<90.5);   priAng=90.5; end;
  if (priAng>179.5)&(priAng<180); priAng=179.5; end;
  if (priAng>=0)&(priAng<0.5);    priAng=0.5; end;
  alfa=priAng;
  if (priAng>45)&(priAng<90); alfa=90-priAng; end;
  if (priAng>90)&(priAng<135); alfa=-(90-priAng); end;
  if (priAng>=135)&(priAng<180); alfa=180-priAng; end;
  radang=priAng*pi/180;
  radpha=alfa*pi/180;
  cpha=cos(radpha);
  spha=sin(radpha);
  tpha=tan(radpha);
  slp=-1/tan(radang);

BB=(priAng<=45)|(priAng>=135) ;
priAng
  if BB
      for sn=1:ns
          if priAng>=135; rl2=nd2/cpha+(ns-sn)*sens-rl1;
          else            rl2=nd2/cpha+(sn-1)*sens-rl1;end;
          
          if priAng<90;   intrcp=rl2/spha+nd2*(1-tpha);
          else            intrcp=-rl2/spha+nd2*(1+tpha);end;
          
     
      
      temp=0;
      for i=1:n
          rl3=(i-0.5-intrcp)/slp+0.5;
          j=round(rl3);
          B= (j>nless)|(j<1);
          
          if ~B; Im=Ph(i,j);Im1=Ph(i,j+1);
              temp=temp+Im-(Im-Im1)*(rl3-j); end; 
      end
      proju(sn,pn)=temp/cpha;
    
      end;% sn
   end ; % BB
   
  
  if ~BB
      for sn=1:ns
          if priAng>=135; rl2=nd2/cpha+(ns-sn)*sens-rl1;
          else            rl2=nd2/cpha+(sn-1)*sens-rl1;end;
          if priAng<90;   intrcp=rl2/spha+nd2*(1-tpha);
          else            intrcp=-rl2/spha+nd2*(1+tpha);end;
          
         temp=0;
      for j=1:n
          rl3=(j-0.5-intrcp)*slp+0.5;
          i=round(rl3);
          
          B= (i>nless)|(i<1);        
          if ~B; Im=Ph(i,j);Im1=Ph(i+1,j);
              temp=temp+Im-(Im-Im1)*(rl3-i); end; 
      end
      proju(sn,pn)=temp/cpha;
      end; % end sn 
  end; % BB

end
% subplot(2,1,2);
% imagesc(proju);
R=proju;