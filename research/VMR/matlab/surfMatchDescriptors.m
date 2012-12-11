function [filenames, points] = surfMatchDescriptors(descriptors, table)
thresh = .08;
num_matches = 0;
filename_table_index = 68;
filenames = cell(1, 1);
dist = zeros(1);
xy = zeros(1, 2);
points = struct('x', 0, 'y', 0);
for index_img = 1 : size(descriptors, 1),
    disp(strcat('processing descriptor ', num2str(index_img), ' of  ', num2str(size(descriptors, 1))));
    d1 = descriptors(index_img, 1 : 64);
    for index_table = 2 : size(table, 1),
       d2 = cell2mat(table(index_table, 1 : 64));
       tmp_dist = distanceSURF(d1, d2);
       if thresh > tmp_dist
           num_matches = num_matches + 1;
           filenames{num_matches, 1} = fullfile(table{index_table, filename_table_index - 1}, table{index_table, filename_table_index});
           dist(num_matches, 1) = tmp_dist;
           xy(num_matches, 1 : 2) = cell2mat(table(index_table, 65 : 66));
       end
    end
end

%[sort_dist, i_dist] = sort(dist, 1);
%filenames = filenames(i_dist, 1);
% disp(filenames);
if isempty(filenames{1, 1}) ~= 1
    [filenames, ~, i_new_names] = unique(filenames);
    for i = 1 : size(filenames, 1),
        points(i).x = xy(i_new_names == i, 1);
        points(i).y = xy(i_new_names == i, 2);
    end
    %disp([points(1).x(:)]);
    %disp([points(1).y(:)]);
    file_counts = accumarray(i_new_names, 1);
    [~, i_new_names] = sort(file_counts, 'descend');
    filenames = filenames(i_new_names);
    points = points(i_new_names);
    %filenames = unique(filenames);
end
end