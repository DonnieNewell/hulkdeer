function trimmed_descriptors = surfGetInteriorDescriptors(descriptors, binaryImage)
%surfGetInteriorDescriptors returns only descriptors that lie within segmented image
trimmed_descriptors = [];
for i = 1 : size(descriptors, 1)
    col = round(descriptors(i, 65));
    row = round(descriptors(i, 66));
    if binaryImage(row, col) == 1
        if isempty(trimmed_descriptors)
            trimmed_descriptors = descriptors(i, :);
        else
            trimmed_descriptors = [trimmed_descriptors; descriptors(i, :)];
        end
    end
end
end