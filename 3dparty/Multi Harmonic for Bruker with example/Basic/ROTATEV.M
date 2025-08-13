function res=rotatev(v,shift)

%shift>0 shift to the left;
%shift<0 shift to the right;

if shift~=0

    L=length(v);
    shift=round(shift);

    vv=0;
    if shift>=0
        xxx=v(1:shift);
        yyy=v(shift+1:L);
        vv=[yyy,xxx];
    else
        xxx=v(L+shift+1:L);
        yyy=v(1:(shift+L));
        vv=[xxx,yyy];
    end
    res=vv;

else
    res=v;
end



% if shift>=0
% for i=1:shift, v(L+i)=v(i); end;
% for i=1:L, vv(i)=v(i+shift); end;
% end;
% if shift<0
% for i=1:shift, v(L+i)=v(i); end;
% for i=1:L, vv(i)=v(i+shift); end;
% end;
% res=vv;
% 
% % for i = 1:l, v(i) = i; end % [1 2 3 4 5, ..];

