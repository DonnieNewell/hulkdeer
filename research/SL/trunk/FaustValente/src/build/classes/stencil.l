HEXFLOATNUM [-]?0x[0-9a-fA-F]\.[0-9a-fA-F]*p[-+][0-9a-fA-F]+
HEXNUM      0x[0-9a-fA-F]+
FLOATNUM    [-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?
INTNUM      [-+]?[0-9]+
ID          [A-Za-z][A-Za-z0-9_]*

%{
  #include <string.h>

  #include "list.h"

  static void comment(void);
  static char *code(void);
  extern unsigned linenum;
  extern int (*get_token)(void);
  extern list includes;
  extern header_t current_header;
%}

%%

\/\*              comment(); /* Start looking for the end of the comment */
\/\/[^\n]*        ;          /* Eat the comment to end of line */
[ \t\v\f\r]       ;          /* Eat all white-space */
\n                linenum++;
^#include         { return INCLUDE_DIRECTIVE_TOK; }
^#define          { return DEFINE_DIRECTIVE_TOK; }
NumDimenstions    { return DIM_TOK; }
StencilSize       { return SIZE_TOK; }
DataType          { return TYPE_TOK; }
FunctionName      { return NAME_TOK; }
ScalarVariables   { return SCALARS_TOK; }
CellValue         { return CELL_TOK; }
EdgeValue         { return EDGE_TOK; }

code_block_out    { return CODE_BLOCK_OUT_TOK; }
code_block_in     { return CODE_BLOCK_IN_TOK; }
^\\/\n            { yytext = code(); return CODELIT_TOK; }
\(                { return OPEN_PAREN_TOK; }
\)                { return CLOSE_PAREN_TOK; }
\[                { return OPEN_SQUAREBRACE_TOK; }
\]                { return CLOSE_SQUAREBRACE_TOK; }
\{                { return OPEN_CURLYBRACE_TOK; }
\}                { return CLOSE_CURLYBRACE_TOK; }
\*                { return STAR_TOK; }
\|                { return BAR_TOK; }
=                 { return EQUALS_TOK; }
,                 { return COMMA_TOK; }
;                 { return SEMI_TOK; }
\".*\"            { return STRINGLIT_TOK; }
{ID}              { return NAME_TOK; }
{INTNUM}          { return INTLIT_TOK; }
{HEXFLOATNUM}     { return FLOATLIT_TOK; }
{HEXNUM}          { return HEXLIT_TOK; }
{FLOATNUM}        { return FLOATLIT_TOK; }
<<EOF>>           { return EOF_TOK; }

%%

unsigned linenum = 1;
char *header_token_strings[] = {
  "INCLUDE_DIRECTIVE_TOK",
  "DEFINE_DIRECTIVE_TOK",
  "ENUM_TOK",
  "CONST_TOK",
  "VOID_TOK",
  "GLFLOAT_TOK",
  "GLDOUBLE_TOK",
  "GLCLAMPF_TOK",
  "GLCLAMPD_TOK",
  "GLBITFIELD_TOK",
  "GLUINT_TOK",
  "GLINT_TOK",
  "GLBOOLEAN_TOK",
  "GLENUM_TOK",
  "GLUSHORT_TOK",
  "GLSHORT_TOK",
  "GLUBYTE_TOK",
  "GLBYTE_TOK",
  "GLUCHAR_TOK",
  "GLCHAR_TOK",
  "GLVOID_TOK",
  "GLSIZEI_TOK",
  "GLASCIISTRING_TOK",
  "GLDISPLAYLIST_TOK",
  "GLIMAGEINDEX_TOK",
  "GLGLSL_ARB_HANDLE_TOK",
  "GLGLSL_PROGRAM_TOK",
  "GLGLSL_SHADER_TOK",
  "CODE_BLOCK_OUT_TOK",
  "CODE_BLOCK_IN_TOK",
  "CODELIT_TOK",
  "GLSHADERINDEX_TOK",
  "GLHANDLE_TOK",
  "OPEN_PAREN_TOK",
  "CLOSE_PAREN_TOK",
  "OPEN_SQUAREBRACE_TOK",
  "CLOSE_SQUAREBRACE_TOK",
  "OPEN_CURLYBRACE_TOK",
  "CLOSE_CURLYBRACE_TOK",
  "STAR_TOK",
  "BAR_TOK",
  "EQUALS_TOK",
  "COMMA_TOK",
  "SEMI_TOK",
  "NAME_TOK",
  "INTLIT_TOK",
  "HEXLIT_TOK",
  "FLOATLIT_TOK",
  "STRINGLIT_TOK",
  "num_header_tokens",
  "TYPE_TOK", /* Not a real token.  Only used to generate an error *
	       * when a type is expected but none was read         */
  "EOF_TOK"
};

char *type_names[] = {
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "void",
  "GLfloat",
  "GLdouble",
  "GLclampf",
  "GLclampd",
  "GLbitfield",
  "GLuint",
  "GLint",
  "GLboolean",
  "GLenum",
  "GLushort",
  "GLshort",
  "GLubyte",
  "GLbyte",
  "GLuchar",
  "GLchar",
  "GLvoid",
  "GLsizei",
  "GLasciistring",
  "GLdisplaylist",
  "GLimageindex",
  "GLGLSL_ARB_handle",
  "GLGLSL_program",
  "GLGLSL_shader",
  "NotAType",
  "NotAType",
  "NotAType",
  "GLshaderindex",
  "GLhandle",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType",
  "NotAType"
};

static void comment(void)
{
  char c1, c2;

  for (;;) {
    while ((c1 = input()) != '*' && c1 != '\0')
      if (c1 == '\n')
	linenum++;

    if ((c2 = input()) != '/' && c2 != '\0') {
      unput(c2);
      continue;
    } else
      break;
  }
}

static char *code(void)
{
  char *s;
  char *s_ptr;
  char c;
  int s_size = 8196;

  c = input(); /* The first character is a garbage newline--eat it */
  linenum++;
  s = s_ptr = malloc(s_size * sizeof (char));

  for (;;) {
    while ((c = input()) != '\\' && c != '\0') {
      if (s_ptr == s + s_size) {
	s = realloc(s, s_size * 2);
	s_ptr = s + s_size;
	s_size *= 2;
      }
      *s_ptr = c;
      s_ptr++;

      /* All code literals are passed through a printf function, so to *
       * get '%', the code needs "%%"                                  */
      if (c == '%') {
	if (s_ptr == s + s_size) {
	  s = realloc(s, s_size * 2);
	  s_ptr = s + s_size;
	  s_size *= 2;
	}
	
	*s_ptr = c;
	s_ptr++;
      }

      if (c == '\n')
	linenum++;
    }

    if (c == '\\' && s_ptr[-1] != '\n') {
      if (s_ptr == s + s_size) {
	s = realloc(s, s_size * 2);
	s_ptr = s + s_size;
	s_size *= 2;
      }
      *s_ptr = c;
      s_ptr++;

      continue;
    } else {
      break;
    }
  }

  *s_ptr = '\0';
  s = realloc(s, s_ptr - s);

  return s;
}

int yywrap()
{
  if (list_not_empty(&includes)) {
    /*   printf("Closing %s\n", current_header.ps.file_name);*/
    strcpy(current_header.ps.file_name,
	   *((header_name_t *) list_item_get_datum(list_get_tail(&includes))));
    fclose(current_header.ps.input);
    current_header.ps.input = yyin = fopen(current_header.ps.file_name, "r");
    linenum = 1;

    /*    printf("Opening %s\n", current_header.ps.file_name);*/

    list_remove_tail(&includes);

    return 0;
  }

  list_delete(&includes);

  return 1;
}
