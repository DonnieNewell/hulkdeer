function csvHOG(file_path, image_path, category)
num_bins = int32(9);
img = imread(image_path);
hog = HOG(img)
descriptors = reshape([hog], num_bins,[]);
descriptors = descriptors'
descriptors = [descriptors repmat(category, size(descriptors, 1), 1)]
dlmwrite(file_path, descriptors, '-append', 'delimiter', ',')
end