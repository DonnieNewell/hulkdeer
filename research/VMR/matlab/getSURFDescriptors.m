function descriptors = getSURFDescriptors(image_path, category)
num_bins = int32(64);
img = imread(image_path);

% get surf descriptors
Options.tresh = 0.0001;
surf = OpenSurf(img, Options);

% convert into row-major cell matrix
descriptors = reshape([surf.descriptor], num_bins,[]);
descriptors = descriptors';
descriptors = [descriptors [surf.x]' [surf.y]'];
descriptors = num2cell(descriptors);

% add category and image path to each record
[~, name, ext] = fileparts(image_path);
category_column = repmat(cellstr(category), size(descriptors, 1), 1);
filename_column = repmat(cellstr(strcat(name,ext)), size(descriptors, 1), 1);
descriptors = cat(2, descriptors, category_column);
descriptors = cat(2, descriptors, filename_column);
end