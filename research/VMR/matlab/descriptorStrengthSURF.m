function [strengths, i_d] = descriptorStrengthSURF(descriptors)
% Donnie Newell (den4gr)
% returns the absolute magnitude of the Haar responses in each region of 
%  the descriptor
stride = 4;

% we are only concerned with 2 absolute sums in each region of descriptor
% since the regions are 4x4 == 2 * 16 = 32
strengths = zeros(size(descriptors, 1), 1);
abs_value_indices = zeros(1,32);
for i = 1 : 16,
    ind = i * stride - 1;
    d_ind = i * 2 - 1;
    abs_value_indices(d_ind) = ind;
    abs_value_indices(d_ind + 1) = ind + 1;   
end
    
for desc_index = 1 : size(descriptors, 1),
    strengths(desc_index, 1) = sum(descriptors(desc_index, abs_value_indices), 2);
end

[strengths, i_d] = sort(strengths, 'descend');
end