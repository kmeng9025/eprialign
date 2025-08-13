% s=0;
% for i=1:10000
%     s=s+  sign( rand(1)-.5);
% end

sum (sign( rand(1,10000000)-.5))