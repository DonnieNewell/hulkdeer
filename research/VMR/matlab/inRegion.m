function region = inRegion(img)
[rows cols channels] = size(img);
region = repmat(uint8(0), rows, cols);
for i = 1 : rows,
   for j = 1 : cols,
      if ((uint8(img(i, j, 1)) ~= 0) && (uint8(img(i, j, 2)) ~= 0) && (uint8(img(i, j, 3)) ~= 0))
          region(i, j) = 1;
      end
   end
end
end