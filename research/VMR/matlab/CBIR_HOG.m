function [filenames, points] = CBIR_HOG(img, table)
thresh = .008;
filename_table_index = 11;
num_matches = 0;
num_windows = 25;
descriptors = HOGMatrix(img);
filenames = cell(5, 1);
distance = zeros(1);
lin_index = zeros(1);
for index_img = 1 : size(descriptors, 1),
    d1 = descriptors(index_img,:);
    for index_table = 1 : size(table, 1),
       d2 = cell2mat(table(index_table, 1:9));
       tmp_dist = distanceHOG(d1, d2);
       if thresh > tmp_dist
           num_matches = num_matches + 1;
           filenames{num_matches, 1} = fullfile(table{index_table, filename_table_index - 1}, table{index_table, filename_table_index});
           distance(num_matches, 1) = tmp_dist;
           lin_index(num_matches) = mod(index_table, num_windows);
       end
    end
end
if isempty(filenames{1, 1}) ~= 1
    [filenames, ~, i_new_names] = unique(filenames);
    points = zeros(size(filenames, 1));
    for i = 1 : size(filenames, 1),
        squares = lin_index(i_new_names == i);
        points(i, 1:length(squares)) = squares;
    end
    file_counts = accumarray(i_new_names, 1);
    [~, i_new_names] = sort(file_counts, 'descend');
    filenames = filenames(i_new_names);
    points = points(i_new_names);
    %filenames = unique(filenames);
end
end