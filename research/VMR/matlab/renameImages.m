function renameImages(directory) 
files = dir(directory);
for i = 1 : length(files),
    
    [path name ext] = fileparts(fullfile(directory, files(i).name));
    from_file = fullfile(directory, files(i).name);
    if files(i).isdir
        continue;
    end
    disp(from_file);
    to_file = fullfile(path, strcat(sprintf('%03d', i),ext));
    disp(to_file);
    
   movefile(from_file, to_file);
end
end