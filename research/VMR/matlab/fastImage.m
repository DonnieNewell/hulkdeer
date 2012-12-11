function fastImage(input_image)
%Make image greyscale
if length(size(input_image)) == 3
	image =  double(input_image(:,:,2));
else
	image = double(input_image);
end

cs = fast_corner_detect_9(image, 30);
c = fast_nonmax(image, 30, cs);

%image(image/4)
imshow(input_image)
axis image
colormap(gray)
hold on
plot(cs(:,1), cs(:,2), 'r.')
plot(c(:,1), c(:,2), 'g.')
e = entropy(input_image)
title_string = strcat('9 point FAST corner detection on an image', ' Entropy = ', num2str(e))
legend('9 point FAST corners', 'nonmax-suppressed corners')
title(title_string)
end