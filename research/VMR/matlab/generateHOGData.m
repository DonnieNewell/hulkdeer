function data = generateHOGData(image_dir)
categories = {'building' 'helicopter' 'ak' 'rpg' 'landscape' };
%categories = {'building'};
extensions = {'*.jpg' '*.png'};
max_images = 50;
data = {};
for cat = 1 : size(categories, 2),
    for ext = 1 : size(extensions, 2),
        images = dir(fullfile(image_dir, categories{cat}, extensions{ext}));
        num_images = min(length(images), max_images);
        for i = 1 : num_images,
            img = fullfile(image_dir, categories{cat}, images(i).name);
            new_desc = getHOGDescriptors(img, categories{cat});
            data = [data; new_desc;];
        end
    end
end
end