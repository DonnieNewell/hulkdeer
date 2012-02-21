// Stencil Language description of simple 2D heated plate.

NumDimensions 2
StencilSize (2, 2)
DataType float
FunctionName runPlate

CellValue {
    return((get(-2, 0) + get(-1, 0) + get(2, 0) + get(1, 0) + get(0, -2) + get(0, -1) + get(0, 2) + get(0, 1)) * 0.25);
}

EdgeValue {
    if (y < 0)
        return(x * 100 / input_size.x);
    else if (y >= input_size.y)
        return(100 - (x * 100 / input_size.x));
    else if (x < 0)
        return(y * 100 / input_size.x);
    else
        return(100 - (y * 100 / input_size.y));
}

