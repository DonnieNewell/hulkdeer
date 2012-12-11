function matches = displayMatches(img, dir, data)
% data_name = 'data.mat';
% category = 'building';
figures_per_row = int32(3);
% data_name = fullfile(dir, category, data_name);
% disp(data_name);
% load(data_name);
num_desc_to_match = 30;
%img = imread(img_name);
[matches points] = CBIR_SURF(img, data, num_desc_to_match);
%[matches, windows] = CBIR_HOG(img, data);
% display results
%disp(matches);
figure;
num_rows = 3;
for i = 1 : min(9, size(matches, 1)),
    image_path = fullfile(dir, matches{i, 1});
    %disp(image_path);
    subplot(3, 3, i);
    hold on;
    img = imread(image_path);
    rows = size(img, 1);
    cols = size(img, 2);
    row_step = idivide(rows, int32(5));
    col_step = idivide(cols, int32(5));
    
    imshow(img);
    
    scatter([points(i).x], [points(i).y], '+');
%    title(strcat(num2str(size(windows(i), 1)),' matches'));
    title(strcat(num2str(size(points(i).x, 1)),' matches'));
    hold off;
end
end