function displayHoughlineMatrix(img, h_mat)
    figure, imshow(img), hold on;
    for i = 1 : size(h_mat, 1),
         plot([h_mat(i, 1); h_mat(i, 3)],[h_mat(i, 2); h_mat(i, 4)],'LineWidth',2,'Color','green');
    end
    hold off;
end