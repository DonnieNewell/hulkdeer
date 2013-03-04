function binary_image = getBinaryImage(queryFilename, fileNames, binaryImages)
%disp(strcat('queryFilename: |', queryFilename, '|'));
%disp('fileNames');
for i = 1 : size(fileNames, 2),
   %disp(strcat('fileName{', num2str(i), '} |', fileNames{i}, '|'));
   if 1 == strcmp(fileNames{i}, queryFilename)
       binary_image = binaryImages{i};
       break;
   end
end
end