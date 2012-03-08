/* -*- Mode: Java; indent-tabs-mode: nil -*-
 *
 * CS 6620 Spring 2010 Term Project
 * Author: Sal Valente <sv9wn@virginia.edu>
 */

import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class Tokenizer
{
public static final int EOF = 0;
public static final int INVALID = 1;
public static final int WORD = 2;

/**
 * Entire file contents.
 */
private char[] contents;

/**
 * Current file contents pointer during parsing.
 */
private int idx;

/**
 * Current file contents line number during parsing.
 */
private int line_number;

/**
 * Read a file into memory so the contents can be tokenized.
 */
public Tokenizer(String filename) throws IOException
{
    int length = (int) new File(filename).length();
    this.contents = new char[length+1];
    FileReader reader = new FileReader(filename);
    int result = reader.read(contents, 0, length);
    reader.close();
    if (result != length)
        throw new IOException(filename + ": only read " + result + " bytes");
    this.contents[length] = 0;
    this.idx = 0;
    this.line_number = 1;
}

/**
 * Get the next token in the file.
 */
public Token getToken()
{
    while (Character.isWhitespace(contents[idx]))
    {
        if (contents[idx] == '\n')
            line_number++;
        idx++;
    }
    char ch = contents[idx];
    if (ch == '/' && contents[idx+1] == '/')
    {
        while (contents[idx] != '\n' && contents[idx] != 0)
            idx++;
        return getToken();
    }
    if (ch == 0)
    {
        return new Token(EOF, null, line_number);
    }
    if (ch == '(' || ch == ')' || ch == '{' || ch == '}' || ch == ',')
    {
        idx++;
        return new Token(ch, null, line_number);
    }
    int start = idx;
    while ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
           (ch >= '1' && ch <= '9') || (ch == '_'))
    {
        idx++;
        ch = contents[idx];
    }
    if (idx > start)
    {
        String value = new String(contents, start, idx-start);
        return new Token(WORD, value, line_number);
    }
    return new Token(INVALID, null, line_number);
}

/**
 * Get a block of C code, assuming there are no closing curly braces
 * in quotes or comments.
 */
public String getBlock()
{
    int count = 0;
    int start = idx;
    while (contents[idx] != 0)
    {
        char ch = contents[idx];
        if (ch == '{')
            count++;
        if (ch == '}')
        {
            if (count == 0)
                break;
            count--;
        }
        idx++;
    }
    // Strip newlines from start and end of block.
    int end = idx;
    while ((end > start) && Character.isWhitespace(contents[end-1]))
        end--;
    int altstart = start;
    while ((altstart < end) && Character.isWhitespace(contents[altstart]))
    {
        if (contents[altstart] == '\n')
        {
            start = altstart+1;
            break;
        }
        altstart++;
    }
    return new String(contents, start, end-start);
}

}
