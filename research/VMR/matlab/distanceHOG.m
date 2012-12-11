function D = distanceHOG(descriptor_1, descriptor_2)
% Donnie Newell
% VMR
D = sum((descriptor_2 - descriptor_1) .^ 2);
end