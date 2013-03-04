function trimmed_data = surfGetInteriorData(data, binaryCellArray, fileNames)
%surfGetInteriorData returns only descriptors that lie within segmented
%   images

file_name = '0';
binaryImage = zeros(1);
trimmed_data = [];
for i = 1 : size(data, 1)
    % get binary image for this fileName
    if 0 == strcmp(file_name, data{i, 68})
        file_name = data{i, 68};
        %disp(file_name);
        for j = 1 : size(fileNames, 2)
            disp(strcat('comaparing ', file_name, ' to ', fileNames{j}));
            if strcmp(file_name, strcat('seg_',fileNames{j}))
                binaryImage = binaryCellArray{j};
                break;
            end
        end
    end
     disp(strcat('descriptor : ', int2str(i)));
     disp(strcat('image : ', file_name));
    col = round(data{i, 65});
    row = round(data{i, 66});
     disp(size(binaryImage));
     disp(size(trimmed_data));
     disp(strcat('binaryImage(', num2str(row), ', ', num2str(col), ')'))
     disp(strcat(' = ', num2str(binaryImage(row, col))));
    if binaryImage(row, col) == 1
        if isempty(trimmed_data)
            trimmed_data = data(i, :);
        else
            trimmed_data = cat(1, trimmed_data, data(i, :));
        end
    end
end
end