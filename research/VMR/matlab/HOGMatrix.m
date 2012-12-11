function mat = HOGMatrix(img)
    num_bins = 9;
    hog = HOG(img);
    mat = reshape([hog], num_bins,[]);
    mat = mat';
end