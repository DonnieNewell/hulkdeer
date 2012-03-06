//

NumDimensions 1
StencilSize (1)
DataType int
FunctionName runCell

ScalarVariables (
    int bornMin, int bornMax, int dieMin, int dieMax
)

CellValue {
    int orig = get(0);
    int sum = 0;
    int i, j, k;
    for (i = -1; i <= 1; i++)
        sum += get(i);
    sum -= orig;
    int retval;
    if(orig>0 && (sum <= dieMax || sum >= dieMin)) retval = 0;
    else if (orig==0 && (sum >= bornMin && sum <= bornMax)) retval = 1;
    else retval = orig;    
    return (retval);
}
