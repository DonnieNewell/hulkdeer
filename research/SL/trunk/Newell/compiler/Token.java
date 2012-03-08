/* -*- Mode: Java; indent-tabs-mode: nil -*-
 *
 * CS 6620 Spring 2010 Term Project
 * Author: Sal Valente <sv9wn@virginia.edu>
 */

public class Token
{
public int id;
public String value;
public int line_number;

public Token(int id, String value, int line_number)
{
    this.id = id;
    this.value = value;
    this.line_number = line_number;
}

}
