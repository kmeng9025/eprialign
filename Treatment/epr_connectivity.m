function res = epr_connectivity(input_mask, the_distance)

the_pattern = ones(3,3,3);

m = input_mask > 0.5;
for ii=1:the_distance
  m = imdilate(m, the_pattern);
end

for ii=1:the_distance
  m = imerode(m, the_pattern);
end

res = sum(input_mask(:)) / sum(m(:));