function x=shiftv1(v,s);
% clc
% clear all
% s=.99;
% t=1:10;
% v=t;

if s==0; x=v; return; end;


up=ceil(abs(s));
down=floor(abs(s));

ss=size(v);
if ss(1)>ss(2); v=v';end;
if s>0
    if down>0
    x=[0*v(1:down) v(1:(end-down))] ;   
    zzz=mean(v(1:down))+0*(1:down);
    x=[zzz  v(1:(end-down))] ;
    else
        x=v;
    end

    r=s-down;
    if r>0
        x=[0 x];
        t=1:length(x);
        t2=t-r+1;
        t2=t2(1:(end-1));
        x2=interp1(t,x,t2);
        x=x2;
    end
end

if s<0
    if down>0
        zzz=mean(v(end-up:end))+0*(1:up);
        x=[v((1+up):end) zzz] ;
    else
        x=v;
    end
    r=abs(s+down);
    if r>0
        x=[x 0];
        t=1:length(x);
        t2=t+r;
        t2=t2(1:(end-1));
        x2=interp1(t,x,t2);
        x=x2;
    end
end

if ss(1)>ss(2); x=x';end;

% plot(v,'-xb'); hold on;
% plot(x,'-xr'); hold off;
% axis tight

%whos x v