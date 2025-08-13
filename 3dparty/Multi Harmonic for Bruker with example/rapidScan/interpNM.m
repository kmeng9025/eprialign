function res=interpNM(sp,M);

N=length(sp);

% if M>N
    t1=1:(N-1)/(N-1):N;
    t2=1:(N-1)/(M-1):N;
    res=interp1(t1,sp,t2)
% else
%     res=sp;
% end
