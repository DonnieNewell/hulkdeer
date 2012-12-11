function generateHOGArffData(image_dir)
data_filename = fullfile(image_dir, 'data.arff');
categories = {'building' 'background' 'motorcycles' 'helicopter' 'ak'...
                'rpg' 'landscape' };
%categories = {'helicopter'};
fileID = fopen(data_filename, 'w');
fprintf(fileID, '@RELATION HOGDescriptors\n');
fprintf(fileID, '@ATTRIBUTE 0-19 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 20-39 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 40-59 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 60-79 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 80-99 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 100-119 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 120-139 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 140-159 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE 160-179 NUMERIC\n');
fprintf(fileID, '@ATTRIBUTE category {');
for attr = 1 : size(categories, 2),
    fprintf(fileID, '%s',categories{attr});
    if attr < size(categories, 2)
        fprintf(fileID, ',');
    end
end
fprintf(fileID, '}\n');
fprintf(fileID, '@ATTRIBUTE file_path STRING\n');
fprintf(fileID, '@DATA\n');
fclose(fileID);
extensions = {'*.jpg' '*.png'};
for cat = 1 : size(categories, 2),
    for ext = 1 : size(extensions, 2),
        images = dir(fullfile(image_dir, categories{cat}, extensions{ext}));
        num_images = length(images);
        for i = 1 : num_images,
            img = fullfile(image_dir, categories{cat}, images(i).name);
            csvHOG(data_filename, img, categories{cat}); 
        end
    end
end
end