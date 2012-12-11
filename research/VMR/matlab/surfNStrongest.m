function [n_strongest, strength] = surfNStrongest(descriptors, n)
% Donnie Newell (den4gr)
% returns the n strongest descriptors 
[strength, index_descriptor] = descriptorStrengthSURF(descriptors);
num_descript = min(n, length(index_descriptor));
n_strongest = descriptors(index_descriptor(1 : num_descript), :);
strength = strength(1 : num_descript);
end