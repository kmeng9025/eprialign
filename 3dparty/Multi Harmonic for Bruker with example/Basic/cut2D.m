function new=cut2D(data2D,cut)

ss=size(data2D); s1=ss(1); s2=ss(2);
new=data2D(cut:(s1-cut),:);
%  slice(v,[5 15 21],21,[1 10])        slice(v,[5 15 21],21,[1 10])       