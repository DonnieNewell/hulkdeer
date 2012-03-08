/* -*- Mode: Java; indent-tabs-mode: nil -*-
 *
 * CS 6620 Spring 2010 Term Project
 * Author: Sal Valente <sv9wn@virginia.edu>
 */

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * This class contains a file with variable names to be substituted from a
 * symbol table.  Also, parts of the template file can be conditionally
 * included or excluded based on whether a symbol is defined.
 */
public class Template
{
private List<String> contents;

public Template(Reader unbufferedReader) throws IOException
{
    BufferedReader reader = new BufferedReader(unbufferedReader);
    this.contents = new ArrayList<String>();
    String line;
    while ((line = reader.readLine()) != null)
        contents.add(line);
    reader.close();
}

public String applySymbolTable(Map<String, String> symbolTable)
{
    StringBuffer buffer = new StringBuffer();
    int ifCount = 0;
    boolean output = true;
    Iterator<String> iterator = contents.iterator();
    while (iterator.hasNext())
    {
        String line = iterator.next();
        if (line.startsWith("#if "))
        {
            String variable = line.substring(4).trim();
            String value = symbolTable.get(variable);
            output = output && (value != null);
            ifCount++;
            continue;
        }
        else if (line.equals("#endif"))
        {
            ifCount--;
            if (ifCount == 0)
            {
                output = true;
                continue;
            }
        }
        if (output)
        {
            buffer.append(replaceSymbols(line, symbolTable));
            buffer.append("\n");
        }
    }
    return buffer.toString();
}

private String replaceSymbols(String line, Map<String, String> symbolTable)
{
    int idx = 0;
    while (true)
    {
        int start = line.indexOf('@', idx);
        if (start < 0)
            break;
        int end = line.indexOf('@', start+1);
        if (end < 0)
            break;
        String variable = line.substring(start+1, end).trim();
        String value = symbolTable.get(variable);
        String prefix = line.substring(0, start);
        String suffix = line.substring(end+1);
        line = prefix;
        if (value != null)
            line += value;
        idx = line.length();
        line += suffix;
    }
    return line;
}

}
