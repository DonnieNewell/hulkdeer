function HOGTest( image_dir )
%HOGTEST applies HOG to all images in directory to match
num_points = 5;
images = dir(fullfile(image_dir, '*.jpg'));
num_images = length(images);
for i = 1 : num_images - 1,
   img1 = fullfile(image_dir, images(i).name);
   for j = i + 1 : min(i + 4, num_images),
      img2 = fullfile(image_dir, images(j).name);
      runHOG(img1, img2, num_points);
   end
end
end