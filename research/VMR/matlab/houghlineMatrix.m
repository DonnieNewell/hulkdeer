function mat = houghlineMatrix(img)
    %num_bins = 4;
    gray_img = rgb2gray(img);
    edge_img = edge(gray_img,'canny');
    [H,T,R] = hough(edge_img);
    P  = houghpeaks(H,10,'threshold',ceil(0.5*max(H(:))));
    lines = houghlines(gray_img, T, R, P, 'FillGap', 5, 'MinLength', 7);
    mat = zeros(6, length(lines));
    for line = 1 : length(lines),
        mat(1,line) = lines(line).point1(1);
        mat(2,line) = lines(line).point1(2);
        mat(3,line) = lines(line).point2(1);
        mat(4,line) = lines(line).point2(2);
        mat(5,line) = lines(line).theta;
        mat(6,line) = lines(line).rho;
    end
    mat = mat';
end
