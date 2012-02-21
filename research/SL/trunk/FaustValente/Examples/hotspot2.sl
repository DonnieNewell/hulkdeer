//

NumDimensions 2
StencilSize (2, 2)
DataType float
FunctionName runHotspot

ScalarVariables (
    float step_div_Cap, float Rx, float Ry, float Rz
)

CellValue {
    float pvalue, value, term1, term2, term3, sum;
    pvalue = read(y * input_size.x + x);
    value = get(0, 0);
    term1 = (get(0, 1) + get(0, -1) - value - value) / Ry;
    term2 = (get(1, 0) + get(-1, 0) - value - value) / Rx;
    term3 = (80.0 - value) / Rz;
    sum = pvalue + term1 + term2 + term3;
    return(value + step_div_Cap * sum);
}
