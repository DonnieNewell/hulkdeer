function mat = SURFMatrix(img)
    num_bins = 64;
    Options.tresh = .0001;
    surf = OpenSurf(img, Options);
    mat = reshape([surf.descriptor], num_bins, []);
    mat = mat';
    mat = [mat [surf.x]' [surf.y]'];
end