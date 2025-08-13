function [Int Hpp]=Lwpp(spectrum,sweep)

sp=zeroLine(spectrum,0.05);
[v i]=min(sp);
[u j]=max(sp);
 N=length(sp);
 Hpp=(i-j)/N*sweep; % pp line-width
 Int=u-v;            % Intensity


