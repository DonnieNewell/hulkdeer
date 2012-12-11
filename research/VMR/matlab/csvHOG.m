function csvHOG(file_path, image_path, category)
num_bins = int32(9);
img = imread(image_path);
hog = HOG(img)
descriptors = reshape([hog], num_bins,[]);
descriptors = descriptors'
%descriptors = [descriptors repmat(category, size(descriptors, 1), 1)]
fileID = fopen(file_path, 'a');
for i = 1:size(descriptors, 1),
    fprintf(fileID,'%f, %f, %f, %f, %f, %f, %f, %f, %f, %s, ''%s''\n', ...
            descriptors(i,1), descriptors(i,2), descriptors(i,3), ...
            descriptors(i,4), descriptors(i,5), descriptors(i,6), ...
            descriptors(i,7), descriptors(i,8), descriptors(i,9), ...
            category, image_path);
end
fclose(fileID);
end