function [sorted_counts ind_counts] = surfCountCoOccur(descriptors)
    thresh = .08;
    counts = zeros(size(descriptors, 1));
    for i = 1 : size(descriptors, 1),
        descriptor1 = descriptors(i, 1 : 64);
        for j = 1 : size(descriptors, 1),
            descriptor2 = descriptors(j, 1 : 64);
            dist = distanceSURF(descriptor1, descriptor2);
            if (dist < thresh) && (dist ~= 0)
                counts(i) = counts(i) + 1;
            end
        end
    end
    [sorted_counts ind_counts] = sort(counts);
end