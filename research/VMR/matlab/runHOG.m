function runHOG(image_path_1, image_path_2, num_points)
%Donnie Newell
%VMIR
img1 = imread(image_path_1);
img2 = imread(image_path_2);
num_bins = int32(9);
num_row_windows = int32(5);
num_col_windows = int32(5);
row_step_1 = size(img1,1) / num_row_windows; 
col_step_1 = size(img1,2) / num_col_windows; 
row_step_2 = size(img2,1) / num_row_windows; 
col_step_2 = size(img2,2) / num_col_windows; 

% run HOG
hog_1 = HOG(img1);
hog_2 = HOG(img2);

% convert to matrix
descriptors_1 = reshape([hog_1], num_bins,[]);
descriptors_2 = reshape([hog_2], num_bins,[]);

% calculate error between HOG windows
error = zeros(1, num_row_windows * num_col_windows);
correlation_1 = 1 : num_row_windows * num_col_windows;
correlation_2 = zeros(1, num_row_windows * num_col_windows);
for i = 1 : num_row_windows * num_col_windows,
       distance = sum((descriptors_2 - repmat(descriptors_1(:, i), [1 num_row_windows * num_col_windows])).^2, 1);
       [error(i), correlation_2(i)] = min(distance);
end

% Sort matches on vector distance
[error, indices] = sort(error);
correlation_1 = correlation_1(indices);
correlation_2 = correlation_2(indices);

% Show both images
height = max(size(img1, 1), size(img2, 1));
width = size(img1, 2) + size(img2, 2);
depth = max(size(img1, 3), size(img2, 3));
disp(size(img1));
disp(size(img2));
img = zeros([height width depth]);
img(1:size(img1, 1), 1:size(img1, 2), :) = img1;
img(1:size(img2, 1), size(img1,2) + 1:size(img1,2) + size(img2, 2), :) = img2;

fig = figure();
set(fig, 'Visible', 'off');
imshow(img / 255); hold on;

% Show the best matches
for i = 1:num_points,
    if abs(error(i)) > .03
        break;
    end
    
    c = rand(1,3);
    % determine which window within each image the rectangles lie.
    window_row_1 = idivide(correlation_1(i) - 1, num_col_windows);
    window_col_1 = mod(correlation_1(i) - 1, num_col_windows);
    window_row_2 = idivide(correlation_2(i) - 1, num_col_windows);
    window_col_2 = mod(correlation_2(i) - 1, num_col_windows);
    
    % determine the actual pixel values in the combined image
    col_1 = window_col_1 * col_step_1;
    row_1 = window_row_1 * row_step_1;
    if size(img1, 1) < size(img2, 1)
        row_1 = row_1 + size(img2, 1) - size(img1, 1);
    end
    col_2 = window_col_2 * col_step_2 + size(img1, 2);
    row_2 = window_row_2 * row_step_2;
    if size(img1, 1) > size(img2, 1)
        row_2 = row_2 + size(img1, 1) - size(img2, 1);
    end
    disp(strcat('window_row_1: ', num2str(window_row_1), ', window_col_1:', num2str(window_col_1)));
    disp(strcat('(col_1:', num2str(col_1), ', row_1:', num2str(row_1),') (col_2:', num2str(col_2),', row_2:', num2str(row_2), ')'));
    rectangle('Position', [col_1, row_1, col_step_1, row_step_1], 'EdgeColor', c);
    rectangle('Position', [col_2, row_2, col_step_2, row_step_2], 'EdgeColor', c);
    %plot([interest_1(correlation_1(i)).x interest_2(correlation_2(i)).x+size(img1_gray,2)], ...
    %        [interest_1(correlation_1(i)).y interest_2(correlation_2(i)).y], '-', 'Color', c);
    %plot([interest_1(correlation_1(i)).x interest_2(correlation_2(i)).x+size(img1_gray,2)], ...
    %    [interest_1(correlation_1(i)).y interest_2(correlation_2(i)).y], 'o', 'Color', c);
end

% print to file
[path1 name1 ext1] = fileparts(image_path_1);
[path2 name2 ext2] = fileparts(image_path_2);
print(fig, '-dpng', fullfile(path1,'output',strcat('hog_', name1, '_', name2, ext1)));
end