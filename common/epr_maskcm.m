function res = epr_maskcm(the_mask)

goodpix=find(the_mask);
[igood,jgood,kgood]=ind2sub(size(the_mask),goodpix);
itgt=mean(igood);
jtgt=mean(jgood);
ktgt=mean(kgood);

res = [itgt,jtgt, ktgt];
