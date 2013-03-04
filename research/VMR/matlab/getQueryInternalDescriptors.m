function descriptors = getQueryInternalDescriptors( queryName, fileNames, binaryImages )
%getQueryInternalDescriptors reads in image and returns the surf
%  descriptors inside the segment
img = imread(queryName);
descriptors = SURFMatrix(img);
[path base ext] = fileparts(queryName);
baseName = strcat(base, ext);
baseName = strrep(baseName, 'seg_', '');
disp(baseName);
binary_image = getBinaryImage(baseName, fileNames, binaryImages);
descriptors = surfGetInteriorDescriptors(descriptors, binary_image);
end