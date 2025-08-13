function filt=RampFilter(N,domain);
%  N=100;
% domain='freq'
switch domain
 case 'freq'
    h=2*(0:N/2)/N;
    filt=[h'; h(end-1:-1:2)'];
    % filt=rotatev(filt',0)';
 case 'time'
    h=zeros(1,N/2);
    h0=1/4;
    t=1:2:(N/2);
    h(t)=-1/pi^2./(t.^2);
    filt=[fliplr(h) h0 h(1:(end-1))];
    %plot(h);
end        

