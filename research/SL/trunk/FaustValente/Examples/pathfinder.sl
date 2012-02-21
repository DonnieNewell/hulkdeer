//

NumDimensions 1
StencilSize (1)
DataType int
FunctionName runPathFinder

CellValue {
    int left = get(-1);
    int up = get(0);
    int right = get(1);
    int shortest = (left < up ? left : up);
    shortest = (shortest < right ? shortest : right);
    int index = input_size.x * (iteration - 1) + x;
    return(shortest + read(index));
}
