/* -*- Mode: Java; indent-tabs-mode: nil -*-
 *
 * CS 6620 Spring 2010 Term Project
 * Author: Sal Valente <sv9wn@virginia.edu>
 */

import java.io.FileWriter;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.util.Map;
import java.util.TreeMap;

public class Tool
{
public static void main(String[] args) throws Exception
{
    new Tool().run(args);
}

public void run(String[] args) throws Exception
{
    String filename, templateFilename, numDimsStr, output;
    Stencil stencil;
    Template template;
    FileWriter writer;
    StringBuffer ht;

    if (args.length != 1)
    {
        System.err.println("Usage: StencilTool filename.sl");
        return;
    }
    filename = args[0];

    stencil = new Stencil();
    if (!stencil.parse(filename))
        return;
    if (!stencil.validate())
        return;
    Map<String, String> symbolTable = new TreeMap<String, String>();
    stencil.writeSymbolTable(symbolTable);

    // Find the template file in the .jar file or in the classpath.
    // Construct the file name depending on the dimensionality of the stencil.
    numDimsStr = symbolTable.get("NumDimensions");
    templateFilename = "template" + numDimsStr + "d.cu";
    InputStream istream = getClass().getResourceAsStream(templateFilename);
    template = new Template(new InputStreamReader(istream));
    output = template.applySymbolTable(symbolTable);

    int idx = filename.lastIndexOf('.');
    if (idx > 0)
        filename = filename.substring(0, idx);
    System.out.println("Writing " + filename + ".cu...");
    writer = new FileWriter(filename + ".cu");
    writer.write(output);
    writer.close();

    // Now we will build up the template for the header.
    // The function call signature depends on the dimensionality.
    
    ht = new StringBuffer();
    ht.append("#ifndef @FunctionName@_H");
    ht.append("#define @FunctionName@_H");
    ht.append("#include <mpi.h>");
    ht.append("#define SL_MPI_Init() int _size_, _rank_; MPI_Init(&argc, &argv); MPI_Comm_size(MPI_COMM_WORLD, &_size_); MPI_Comm_rank(MPI_COMM_WORLD, &_rank_)");
    ht.append("#define SL_PMI_Finalize() MPI_Finalize()");
    ht.append("void @FunctionName@SetData(@DataType@ *, int);\n");
    ht.append("void @FunctionName@(@DataType@ *, ");
    ht.append("#endif");

    int numDims = Integer.parseInt(numDimsStr);
    for (int i = 0; i < numDims; i++)
        ht.append("int, ");
    ht.append("int @ScalarVariables@);\n");
    template = new Template(new StringReader(ht.toString()));
    output = template.applySymbolTable(symbolTable);

    System.out.println("Writing " + filename + ".h...");
    writer = new FileWriter(filename + ".h");
    writer.write(output);
    writer.close();
}

}
