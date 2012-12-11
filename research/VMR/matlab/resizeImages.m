function resizeImages(folder, regex, scale)
    file_list = dir(fullfile(folder, regex));
    disp(strcat(regex, ' matched ', num2str(length(file_list)), ' files.'));
    for i = 1 : length(file_list),
        old_name = file_list(i).name;
        full_name = fullfile(folder, old_name);
        [path, name, ext] = fileparts(full_name);
        disp(full_name);
        img = imread(full_name);
        info = imfinfo(full_name);
        img = imresize(img, scale);
        new_height = scale * info.Height;
        new_width = scale * info.Width;
        new_name = strcat(path, '\..\', name, '_', num2str(new_width), 'x', num2str(new_height), ext);
        disp(strcat('new_file: ', new_name));
        imwrite(img, new_name, info.Format);
    end
end