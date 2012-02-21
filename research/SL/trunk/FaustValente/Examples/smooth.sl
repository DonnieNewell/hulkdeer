NumDimensions 3
StencilSize (2, 2, 2)
DataType float
FunctionName runSmooth

CellValue {
  int i, j, k;
  int sum;

  for (sum = 0, i = -2; i < 2; i++) {
    for (j = -2; j < 2; j++) {
      for (k = -2; k < 2; k++) {
        sum += get(i, j, k);
      }
    }
  }
}

EdgeValue {
  return value;
}

CellDataType float
CellScalarVariables (float epsilon)
ConvergeValue {
  return (abs(get(0, 0, 0) - getNew(0, 0, 0)) < epsilon);
}