function csvSURF(file_path, image_path, category)
num_bins = int32(64);
img = imread(image_path);
Options.tresh = 0.0001;
surf = OpenSurf(img)
descriptors = reshape([surf.descriptor], num_bins,[]);
descriptors = descriptors'
%descriptors = [descriptors repmat(category, size(descriptors, 1), 1)]
fileID = fopen(file_path, 'a');
for i = 1:size(descriptors, 1),
    for d = 1 : num_bins,
        fprintf(fileID, '%f, ', descriptors(i, d));
    end
    fprintf(fileID, '%s, %s\n', category, image_path);
end
fclose(fileID);
end