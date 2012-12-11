function descriptors = getHOGDescriptors(image_path, category)
num_bins = int32(9);
img = imread(image_path);

% get descriptors
descriptors = HOGMatrix(img);

descriptors = num2cell(descriptors);

% add category and image path to each record
[~, name, ext] = fileparts(image_path);
category_column = repmat(cellstr(category), size(descriptors, 1), 1);
filename_column = repmat(cellstr(strcat(name,ext)), size(descriptors, 1), 1);
descriptors = cat(2, descriptors, category_column);
descriptors = cat(2, descriptors, filename_column);
end