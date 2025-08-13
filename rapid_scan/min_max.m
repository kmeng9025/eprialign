function [min_A max_A max_h min_h mid_h]=min_max(h,A) 
[max_A max_i]=max(A);
[min_A min_i]=min(A);

max_h=h(max_i);
min_h=h(min_i);
mid_h=mean([min_h max_h]);
