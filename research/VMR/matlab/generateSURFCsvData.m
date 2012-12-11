function generateSURFCsvData(image_dir)
categories = {'building' '102.helicopter-101' '001.ak47' '002.american-flag'};
%categories = {'102.helicopter-101'};
extensions = {'*.jpg' '*.png'};
max_images_per_category = 50;
for category = 1 : size(categories, 2),
    disp(categories);
    disp(size(categories));
    data_filename = fullfile(image_dir, strcat(categories{1, category}, '.csv'));
    fileID = fopen(data_filename, 'w');
    fprintf(fileID, 'd1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, ');
    fprintf(fileID, 'd17, d18, d19, d20, d21, d22, d23, d24, d25, d26, d27, d28, d29, d30, d31, d32, ');
    fprintf(fileID, 'd33, d34, d35, d36, d37, d38, d39, d40, d41, d42, d43, d44, d45, d46, d47, d48, ');
    fprintf(fileID, 'd49, d50, d51, d52, d53, d54, d55, d56, d57, d58, d59, d60, d61, d62, d63, d64, ');
    fprintf(fileID, 'category, file_name\n');
    fclose(fileID);
    for ext = 1 : size(extensions, 2),
        disp(category);
        images = dir(fullfile(image_dir, categories{1, category}, extensions{ext}));
        num_images = length(images);
        if num_images > max_images_per_category
            num_images = max_images_per_category;
        end
        for i = 1 : num_images,
            img = fullfile(image_dir, categories{1, category}, images(i).name);
            csvSURF(data_filename, img, categories{1, category}); 
        end
    end
end
end