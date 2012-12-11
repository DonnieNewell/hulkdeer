function runSURF(image_path_1, image_path_2, num_points)
%Donnie Newell
%VMIR
img1 = imread(image_path_1);
img2 = imread(image_path_2);
%convert to grayscale
img1_gray = rgb2gray(img1);
img2_gray = rgb2gray(img2);
%img1_gray = img1;
%img2_gray = img2;

% Perform Histogram Equalization to enhance contrast
%img1_gray = histeq(img1_gray, 128);
%img2_gray = histeq(img2_gray, 128);
% Get the Key Points
%Options.upright = true;
Options.tresh = 0.0001;
interest_1 = OpenSurf(img1_gray,Options);
interest_2 = OpenSurf(img2_gray,Options);
% Put the landmark descriptors in a matrix
descriptors_1 = reshape([interest_1.descriptor],64,[]);
descriptors_2 = reshape([interest_2.descriptor],64,[]);
% Find the best matches
error = zeros(1, length(interest_1));
correlation_1 = 1:length(interest_1);
correlation_2 = zeros(1, length(interest_1));
for i = 1:length(interest_1),
    distance = sum((descriptors_2 - repmat(descriptors_1(:, i), [1 length(interest_2)])).^2, 1);
    [error(i), correlation_2(i)] = min(distance);
end
% Sort matches on vector distance
[error, indices] = sort(error);
correlation_1 = correlation_1(indices);
correlation_2 = correlation_2(indices);
% Show both images
height = max(size(img1_gray, 1), size(img2_gray, 1));
width = size(img1_gray, 2) + size(img2_gray, 2);
depth = max(size(img1, 3), size(img2, 3));
disp(size(img1_gray));
disp(size(img2_gray));
img = zeros([height width depth]);
img(1:size(img1_gray, 1), 1:size(img1_gray, 2), :) = img1;
img(1:size(img2_gray, 1), size(img1_gray,2) + 1:size(img1_gray,2) + size(img2_gray, 2), :) = img2;
%PaintSURF(img1, interest_1);
%figure, imshow(img1);
%PaintSURF(img2, interest_2);
%figure, imshow(img2);
fig = figure();
set(fig, 'Visible', 'off');
imshow(img / 255); hold on;
% Show the best matches
for i = 1:num_points,
    if abs(error(i)) > .015
        break;
    end
    c = rand(1,3);
    plot([interest_1(correlation_1(i)).x interest_2(correlation_2(i)).x+size(img1_gray,2)],[interest_1(correlation_1(i)).y interest_2(correlation_2(i)).y],'-','Color',c)
    plot([interest_1(correlation_1(i)).x interest_2(correlation_2(i)).x+size(img1_gray,2)],[interest_1(correlation_1(i)).y interest_2(correlation_2(i)).y],'o','Color',c)
end

% print to file
[path1 name1 ext1] = fileparts(image_path_1);
[path2 name2 ext2] = fileparts(image_path_2);
print(fig, '-dpng', fullfile(path1,'output',strcat('corr_', name1, '_', name2, ext1)));
end