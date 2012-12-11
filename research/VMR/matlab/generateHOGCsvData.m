function generateHOGCsvData(image_dir)
data_filename = fullfile(image_dir, 'data.csv');
categories = {'building' 'helicopter' 'ak' 'rpg' 'landscape' };
%categories = {'helicopter'};
fileID = fopen(data_filename, 'w');
fprintf(fileID, '0-19, 20-39, 40-59, 60-79, 80-99, 100-119, 120-139, 140-159, 160-179, category, file_path\n');
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