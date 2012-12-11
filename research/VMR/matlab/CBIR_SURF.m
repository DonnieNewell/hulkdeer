function [filenames, points] = CBIR_SURF(img, table, num_descriptors)
filename_table_index = 68;
num_cooccur_imgs = 3;
num_cooccur_desc = 15;
descriptors = SURFMatrix(img);
descriptors = surfNStrongest(descriptors, num_descriptors);

[filenames points] = surfMatchDescriptors(descriptors, table);

% find which descriptors occur the most in the results set
% name_vector = table(:, filename_table_index);
% 
% matching = zeros(1, 66);
% counts = zeros(1,1);
% current = 1;
% for i = 1 : num_cooccur_imgs,
%     [path file ext] = fileparts(filenames{i});
%     img_name = [file ext];
%     disp(strcat('now comparing cooccurance image ', img_name));
%     matching_row_ind = strcmp(name_vector, img_name) == 1;
%     matches = cell2mat(table(matching_row_ind, 1 : 66));
%     strongest = surfNStrongest(matches, 10);
%     disp('strongest size');
%     disp(size(strongest, 1));
%     disp('current : ');
%     disp(current);
%     matching(current : current + size(strongest, 1) - 1, :) = strongest;
%     current = current + size(strongest, 1);
% end
% [cooccur_counts ind_cooccur] = surfCountCoOccur(matching);
% matching = matching(ind_cooccur, :);
% 
% % search using the top N cooccurring descriptors from
% % the initial result set.
% [filenames points] = surfMatchDescriptors(matching(1 : num_cooccur_desc, :), table);
% 
% disp(points);
end