function data = generateSURFData(image_dir)
%categories = {'building' 'helicopter' 'ak' 'rpg' 'landscape' };
categories = {'building' 'ak47' 'mix' 'rpg'};
extensions = {'*.jpg' '*.png'};
max_images = 50;
data = {};
for cat = 1 : size(categories, 2),
    for ext = 1 : size(extensions, 2),
        img_regex = fullfile(image_dir, categories{cat}, extensions{ext});
        images = dir(img_regex);
        
        % reduce time to test
        num_images = min(size(images, 1), max_images);
        for i = 1 : num_images,
            img = fullfile(image_dir, categories{cat}, images(i).name);
            new_desc = getSURFDescriptors(img, categories{cat});
            data = [data; new_desc;];
        end
    end
end
end