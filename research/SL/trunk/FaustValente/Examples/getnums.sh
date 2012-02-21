echo Collecting $1 numbers for $2 machine 

outfile=PPoPPCell$1$2.Nums.txt
echo Generating $outfile
./cell 100 100 100 1 1 2>$outfile

outfile=PPoPPHotSpot$1$2.Nums.txt
echo Generating $outfile
./hotspot 2000 1 1 2>$outfile

outfile=PPoPPPathfinder$1$2.Nums.txt
echo Generating $outfile
./pathfinder 1000000 1 1 2>$outfile

outfile=PPoPPPlate$1$2.Nums.txt
echo Generating $outfile
./plate 2000 2000 1 1 2>$outfile

cp plate.cu platetemp.cu

echo Compiling platehalo
cp platehalo.cu plate.cu
make plate 2>/dev/null
outfile=PPoPPPlateHalo$1$2.Nums.txt
echo Generating $outfile
./plate 2000 2000 1 1 2>$outfile

echo Compiling platePP
cp platePP.cu plate.cu
make plate 2>/dev/null
outfile=PPoPPPlatePP$1$2.Nums.txt
echo Generating $outfile
./plate 2000 2000 1 1 2>$outfile

echo ReCompiling plate
cp platetemp.cu plate.cu
make plate 2>/dev/null
