function [ Output, Seed] = Pick_numbers_from_bag( Vec, Reps , varargin  )

%Function to randomize groups of experiments while maintaining that we
%never to the second animal of group 2 before the first animal of group 3.
%Vec is a vector of the possible values to choose from.
%Reps is how many of each experiement you want at the end. varagin should
%be an rng seed produced by matlab. You can save these like variables to
%make this reproducible.
%-MM

Output = [];
if  ~(length(varargin) >2)
    Seed = rng;
end

rng(Seed);

for ii = 1:Reps;

Bag = Vec;
for  jj=1:length(Vec)
    roll = randi(length(Bag),1);
    Output(end+1) = Bag(roll);
    Bag(roll) = [];
end

end
Output = Output';
end

