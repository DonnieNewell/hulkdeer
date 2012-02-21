/* -*- Mode: Java; indent-tabs-mode: nil -*-
 *
 * CS 6620 Spring 2010 Term Project
 * Author: Sal Valente <sv9wn@virginia.edu>
 */

import java.io.IOException;
import java.util.ArrayList;
/*import java.util.Iterator;*/
import java.util.List;
import java.util.Map;

/**
 * This class contains a complete description of a Stencil calculation.
 */
public class Stencil
{
private int numDimensions;
private int[] stencilSize;
private String dataType;
private String functionName;
private List<String> scalarTypes;
private List<String> scalarNames;
private String cellValue;
private String edgeValue;
    
/**
 * Read a stencil language file into a Stencil description object.
 */
public boolean parse(String filename) throws IOException
{
    Tokenizer tokenizer = new Tokenizer(filename);
    while (true)
    {
        Token token = tokenizer.getToken();
        if (token.id == Tokenizer.EOF)
            break;
        if (!process(filename, tokenizer, token))
        {
            System.err.println(filename + ":" + token.line_number +
                               ": Parse error");
            return false;
        }
    }
    return true;
}

private boolean process(String filename, Tokenizer tokenizer, Token token)
{
    if (token.id != Tokenizer.WORD)
    {
        return false;
    }
    String value = token.value;
    if (value.equalsIgnoreCase("NumDimensions"))
    {
        token = tokenizer.getToken();
        try
        {
            if (token.id == Tokenizer.WORD)
                this.numDimensions = Integer.parseInt(token.value);
        }
        catch (Exception ignore) {}
        return((numDimensions > 0) && (numDimensions <= 3));
    }
    if (value.equalsIgnoreCase("StencilSize"))
    {
        token = tokenizer.getToken();
        if (token.id != '(')
            return false;
        List<Integer> list = new ArrayList<Integer>();
        while (true)
        {
            token = tokenizer.getToken();
            if (token.id != Tokenizer.WORD)
                return false;
            try
            {
                list.add(new Integer(token.value));
            }
            catch (Exception exception)
            {
                return false;
            }
            token = tokenizer.getToken();
            if (token.id == ')')
                break;
            if (token.id != ',')
                return false;
        }
        this.stencilSize = new int[list.size()];
        for (int i = 0; i < stencilSize.length; i++)
            stencilSize[i] = list.get(i).intValue();
        return true;
    }
    if (value.equalsIgnoreCase("DataType"))
    {
        token = tokenizer.getToken();
        if (token.id != Tokenizer.WORD)
            return false;
        this.dataType = token.value;
        return true;
    }
    if (value.equalsIgnoreCase("FunctionName"))
    {
        token = tokenizer.getToken();
        if (token.id != Tokenizer.WORD)
            return false;
        this.functionName = token.value;
        return true;
    }
    if (value.equalsIgnoreCase("ScalarVariables"))
    {
        this.scalarTypes = new ArrayList<String>();
        this.scalarNames = new ArrayList<String>();
        token = tokenizer.getToken();
        if (token.id != '(')
            return false;
        while (true)
        {
            token = tokenizer.getToken();
            if (token.id != Tokenizer.WORD)
                return false;
            scalarTypes.add(token.value);
            token = tokenizer.getToken();
            if (token.id != Tokenizer.WORD)
                return false;
            scalarNames.add(token.value);
            token = tokenizer.getToken();
            if (token.id == ')')
                break;
            if (token.id != ',')
                return false;
        }
        return true;
    }
    if (value.equalsIgnoreCase("CellValue"))
    {
        token = tokenizer.getToken();
        if (token.id != '{')
            return false;
        this.cellValue = tokenizer.getBlock();
        token = tokenizer.getToken();
        return(token.id == '}');
    }
    if (value.equalsIgnoreCase("EdgeValue"))
    {
        token = tokenizer.getToken();
        if (token.id != '{')
            return false;
        this.edgeValue = tokenizer.getBlock();
        token = tokenizer.getToken();
        return(token.id == '}');
    }

    return false;
}

/**
 * Verify that the stencil description is complete and consistent.
 */
public boolean validate()
{
    if (numDimensions == 0)
    {
        System.err.println("Error: Missing NumDimensions");
        return false;
    }
    if ((stencilSize == null) || (stencilSize.length != numDimensions))
    {
        System.err.println("Error: Missing or invalid StencilSize");
        return false;
    }
    if (dataType == null)
    {
        System.err.println("Error: Missing DataType");
        return false;
    }
    if (functionName == null)
    {
        System.err.println("Error: Missing FunctionName");
        return false;
    }
    if (cellValue == null)
    {
        System.err.println("Error: Missing CellValue");
        return false;
    }
    // ScalarVariables and EdgeValue are optional.
    return true;
}

/**
 * Write this stencil description into a symbol table.
 */
public void writeSymbolTable(Map<String, String> symbolTable)
{
    symbolTable.put("NumDimensions", "" + numDimensions);
    String text = "(" + stencilSize[0];
    for (int i = 1; i < stencilSize.length; i++)
        text = text + "," + stencilSize[i];
    text = text + ")";
    symbolTable.put("StencilSize", text);
    symbolTable.put("DataType", dataType);
    symbolTable.put("FunctionName", functionName);
    if (scalarTypes != null)
    {
        String types = "", names = "";
        int size = scalarTypes.size();
        for (int i = 0; i < size; i++)
        {
            String name = scalarNames.get(i);
            types += ", " + scalarTypes.get(i) + " " + name;
            names += ", " + name;
        }
        symbolTable.put("ScalarVariables", types);
        symbolTable.put("ScalarVariableNames", names);
    }
    symbolTable.put("CellValue", cellValue);
    symbolTable.put("EdgeValue", edgeValue);
}

}
