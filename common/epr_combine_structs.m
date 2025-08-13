function s = epr_combine_structs(s, s2)

f = fieldnames(s2);
for ii=1:length(f)
  s.(f{ii})=s2.(f{ii});
end