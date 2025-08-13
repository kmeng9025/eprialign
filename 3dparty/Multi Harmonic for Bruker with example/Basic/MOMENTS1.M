function [momentleft,momentright]=moments1(m);
% m= [ 2 0 0 0 0 0 0];
% делит вектор попалам, надодит в каждой половине "момент силы" =
% Integrate[f(x) x dx]
l=length(m);cntr=round(l/2);
if l/2-cntr==0 % even 
 for i = 1:cntr, vy(i) = (2*i-1)/2; end    % [1 2 3 4 5, ..]
 vx=fliplr(vy);
 xxx=m(1:cntr); 
 momentleft=dot(xxx,vx) ;
 yyy=m(cntr+1:l);
 momentright=dot(yyy,vy) ;
 else % odd length of vector
 cntr1=(l-1)/2;
 for i = 1:cntr1, vy(i) = i; end    % [1 2 3 4 5, ..]
 vx=fliplr(vy);
 xxx=m(1:cntr1); 
 momentleft=dot(xxx,vx) ;
 yyy=m(cntr1+2:l);
 momentright=dot(yyy,vy) ;   
end

% shft=round(l/2 -moment1);
% nw0=rotatev(x,-shft);
% nw1=rotatev(x,-shft+1);
% nw2=rotatev(x,-shft+2);
% nw_1=rotatev(x,-shft-1);
% nw_2=rotatev(x,-shft-2);








% clear all;
% m=[ 1 2 3 3 2 1; 4 5  6 6 5 4];
% k= 1;
% x= m(k,:);                    % k row of matrix m
% l=length(x); 
% for i = 1:l, v(i) = (2*i-1)/2; end    % [1 2 3 4 5, ..]
% r=dot(x,v); s=sum(x);
% moment1=r/s;           % 1st moment
% shft=floor(l/2 -moment1);
% newvector=rotatev(x,shft)
