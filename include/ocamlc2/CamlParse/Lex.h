#pragma once

#include <cctype>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "camlparse-lex"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {
inline namespace CamlParse {

/// Structure definition for a location in the OCaml parse tree output
struct LexLocation {
  std::shared_ptr<std::string> file; ///< filename
  int startLine;                     ///< start line number
  int startCol;                      ///< start column number
  int endLine;                       ///< end line number
  int endCol;                        ///< end column number
};

/// Token types returned by the lexer
enum Token : int {
  // Single character tokens use their ASCII value
  tok_bracket_open = '[',
  tok_bracket_close = ']',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_angle_open = '<',
  tok_angle_close = '>',
  tok_comma = ',',
  tok_dot = '.',
  tok_colon = ':',
  tok_plus = '+',
  tok_minus = '-',
  tok_equal = '=',

  // Special tokens
  tok_eof = -1,
  tok_error = -2,
  
  // Identifiers and literals
  tok_identifier = -3,
  tok_string_literal = -4,
  tok_integer_literal = -5,
  tok_float_literal = -6,
  
  // Special OCaml AST tokens
  tok_structure_item = -7,
  tok_pattern = -8,
  tok_expression = -9,
  tok_def = -10,
  tok_location = -11,
  tok_pstr_value = -12,
  tok_ppat_var = -13,
  tok_pexp_constant = -14,
  tok_pconst_int = -15,
  tok_pconst_string = -16,
  tok_pconst_float = -17,
  tok_ppat_constant = -18,
  tok_none = -19,
  tok_some = -20,
  tok_ghost = -21,
  tok_rec = -22,
  tok_nonrec = -23,
  tok_pexp_apply = -24,
  tok_pexp_ident = -25,
  tok_arg = -26,
  tok_nolabel = -27,
  tok_pexp_function = -28,
  tok_pparam_val = -29,
  tok_pfunction_body = -30,
  
  // Special composite tokens
  tok_def_tag_open = -100,  // <def>
  tok_case = -102,          // <case>
  tok_arg_tag = -103,       // <arg>
  tok_when = -104,          // <when>
};

inline constexpr llvm::StringRef getTokenName(Token tok) {
  switch (tok) {
  case tok_bracket_open: return "bracket_open";
  case tok_bracket_close: return "bracket_close";
  case tok_parenthese_open: return "parenthese_open";
  case tok_parenthese_close: return "parenthese_close";
  case tok_angle_open: return "angle_open";
  case tok_angle_close: return "angle_close";
  case tok_comma: return "comma";
  case tok_dot: return "dot";
  case tok_colon: return "colon";
  case tok_plus: return "plus";
  case tok_minus: return "minus";
  case tok_equal: return "equal";

  // Special tokens
  case tok_eof: return "eof";
  case tok_error: return "error";
  
  // Identifiers and literals
  case tok_identifier: return "identifier";
  case tok_string_literal: return "string_literal";
  case tok_integer_literal: return "integer_literal";
  case tok_float_literal: return "float_literal";
  
  // Special OCaml AST tokens
  case tok_structure_item: return "structure_item";
  case tok_pattern: return "pattern";
  case tok_expression: return "expression";
  case tok_def: return "def";
  case tok_location: return "location";
  case tok_pstr_value: return "pstr_value";
  case tok_ppat_var: return "ppat_var";
  case tok_pexp_constant: return "pexp_constant";
  case tok_pconst_int: return "pconst_int";
  case tok_pconst_string: return "pconst_string";
  case tok_pconst_float: return "pconst_float";
  case tok_ppat_constant: return "ppat_constant";
  case tok_none: return "none";
  case tok_some: return "some";
  case tok_ghost: return "ghost";
  case tok_rec: return "rec";
  case tok_nonrec: return "nonrec";
  case tok_pexp_apply: return "pexp_apply";
  case tok_pexp_ident: return "pexp_ident";
  case tok_arg: return "arg";
  case tok_nolabel: return "nolabel";
  case tok_pexp_function: return "pexp_function";
  case tok_pparam_val: return "pparam_val";
  case tok_pfunction_body: return "pfunction_body";
  
  // Special composite tokens
  case tok_def_tag_open: return "<def>";
  case tok_case: return "<case>";
  case tok_arg_tag: return "<arg>";
  case tok_when: return "<when>";
  }
}

/// Lexer for OCaml parse tree output
/// This lexer reads the output of `ocamlc -dparsetree file.ml` and tokenizes it.
class Lexer {
public:
  /// Create a lexer for the given filename.
  Lexer(std::string filename)
      : lastLocation({std::make_shared<std::string>(std::move(filename)), 
                      0, 0, 0, 0}) {}
  virtual ~Lexer() = default;

  /// Get the current token in the stream
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it
  Token getNextToken() { return curTok = getTok(); }

  /// Consume the current token, asserting it matches the expected token
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() is an identifier)
  llvm::StringRef getIdentifier() {
    assert(curTok == tok_identifier || 
           (curTok >= tok_structure_item && curTok <= tok_nonrec) ||
           curTok == tok_case || curTok == tok_arg_tag || curTok == tok_when);
    DBGS("got identifier: " << identifierStr << "\n");
    return identifierStr;
  }

  /// Return the current string literal (prereq: getCurToken() == tok_string_literal)
  llvm::StringRef getStringLiteral() {
    assert(curTok == tok_string_literal);
    DBGS("got string literal: " << stringLiteral << "\n");
    return stringLiteral;
  }

  /// Return the current integer literal (prereq: getCurToken() == tok_integer_literal)
  int64_t getIntegerLiteral() {
    assert(curTok == tok_integer_literal);
    DBGS("got integer literal: " << integerLiteral << "\n");
    return integerLiteral;
  }

  /// Return the current float literal (prereq: getCurToken() == tok_float_literal)
  double getFloatLiteral() {
    assert(curTok == tok_float_literal);
    DBGS("got float literal: " << floatLiteral << "\n");
    return floatLiteral;
  }

  /// Return the current location
  LexLocation getLastLocation() { return lastLocation; }

  /// Return the current line in the file
  int getLine() { return curLineNum; }

  /// Return the current column in the file
  int getCol() { return curCol; }

protected:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to end with "\n"
  virtual llvm::StringRef readNextLine() = 0;

  /// Get the next character from the input
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    char nextchar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  /// Look ahead at the next character without consuming it
  int peekNextChar() {
    if (curLineBuffer.empty())
      return EOF;
    return curLineBuffer.front();
  }

  void eatChar(char c) {
    lastChar = getNextChar();
    assert(lastChar == c && "eatChar: char mismatch");
  }

  /// Skip whitespace characters
  void skipWhitespace() {
    while (isspace(lastChar))
      lastChar = getNextChar();
  }

  /// Check for a specific tag like "<def>" or "<arg>"
  bool checkForTag(Token& resultToken) {
    TRACE();
    // Already consumed the '<'
    std::string tag;
    
    // Read tag name
    while (isalnum(peekNextChar()) || peekNextChar() == '_') {
      tag += getNextChar();
    }
    DBGS("got tag: " << tag << "\n");
    
    // Check if it's our target tag
    const auto tagToken = (tag == "def")    ? tok_def
                          : (tag == "case") ? tok_case
                          : (tag == "arg")  ? tok_arg
                          : (tag == "when") ? tok_when
                                            : std::optional<Token>{};
    if (tagToken && peekNextChar() == '>') {
      DBGS("got '>' with tag: " << tag << "\n");
      eatChar('>');
      
      // Return the composite token
      resultToken = tagToken.value();
      return true;
    } else {
      DBGS("no '>' with tag: " << tag << "\n");
    }
    return false;
  }
  
  // Helper to map tag names to token offsets
  int getTagTokenOffset(const std::string& tagName) {
    if (tagName == "def") return 0;
    if (tagName == "case") return 1;
    if (tagName == "arg") return 2;
    if (tagName == "when") return 3;
    return 99; // Unknown tag
  }

  /// Parse and return the next token
  Token getTok() {
    TRACE();
    // Skip any whitespace
    skipWhitespace();

    // Save the current location before reading the token characters
    lastLocation.startLine = curLineNum;
    lastLocation.startCol = curCol;

    // Check for end of file
    if (lastChar == EOF)
      return tok_eof;

    // Check for special tags like <def>, <case>, etc.
    if (lastChar == '<') {
      DBGS("got '<'\n");
      
      // Try to recognize special tags
      Token tagToken;
      bool isTag = checkForTag(tagToken);
      DBGS("isTag: " << isTag << "\n");
      DBGS("tagToken: " << tagToken << "\n");
                   
      if (isTag) {
        DBGS("returning tagToken: " << tagToken << "\n");
        lastChar = getNextChar(); // Consume '>'
        return tagToken;
      }
      
      // If we couldn't recognize a special tag, return the '<' token
      // The tag content will be handled as an identifier in the next token
      return tok_angle_open;
    }

    // Check for brackets, parentheses, and other single-char tokens
    if (lastChar == '[' || lastChar == ']' || lastChar == '(' || lastChar == ')' ||
        lastChar == '>' || lastChar == ',' || lastChar == '.' ||
        lastChar == ':' || lastChar == '+' || lastChar == '-' || lastChar == '=') {
      Token thisChar = static_cast<Token>(lastChar);
      lastChar = getNextChar();
      return thisChar;
    }

    // Identifier: [A-Za-z_][A-Za-z0-9_]*
    if (isalpha(lastChar) || lastChar == '_') {
      identifierStr = "";
      do {
        identifierStr += static_cast<char>(lastChar);
        lastChar = getNextChar();
      } while (isalnum(lastChar) || lastChar == '_' || lastChar == '.');

      // Check for OCaml AST keywords
      if (identifierStr == "structure_item") return tok_structure_item;
      if (identifierStr == "pattern") return tok_pattern;
      if (identifierStr == "expression") return tok_expression;
      if (identifierStr == "def") return tok_def;
      if (identifierStr == "Pstr_value") return tok_pstr_value;
      if (identifierStr == "Ppat_var") return tok_ppat_var;
      if (identifierStr == "Pexp_constant") return tok_pexp_constant;
      if (identifierStr == "PConst_int") return tok_pconst_int;
      if (identifierStr == "PConst_string") return tok_pconst_string;
      if (identifierStr == "PConst_float") return tok_pconst_float;
      if (identifierStr == "Ppat_constant") return tok_ppat_constant;
      if (identifierStr == "None") return tok_none;
      if (identifierStr == "Some") return tok_some;
      if (identifierStr == "ghost") return tok_ghost;
      if (identifierStr == "Rec") return tok_rec;
      if (identifierStr == "Nonrec") return tok_nonrec;
      if (identifierStr == "Pexp_apply") return tok_pexp_apply;
      if (identifierStr == "Pexp_ident") return tok_pexp_ident;
      if (identifierStr == "arg") return tok_arg;
      if (identifierStr == "Nolabel") return tok_nolabel;
      if (identifierStr == "Pexp_function") return tok_pexp_function;
      if (identifierStr == "Pparam_val") return tok_pparam_val;
      if (identifierStr == "Pfunction_body") return tok_pfunction_body;
      
      // Handle some special case-sensitive keywords
      if (identifierStr == "case") return tok_case;
      if (identifierStr == "when") return tok_when;

      return tok_identifier;
    }

    // String: "..."
    if (lastChar == '"') {
      stringLiteral = "";
      lastChar = getNextChar(); // Skip opening quote

      // Read the string content
      while (lastChar != '"' && lastChar != EOF) {
        // Handle escaping within strings
        if (lastChar == '\\') {
          lastChar = getNextChar(); // Skip the backslash
          switch (lastChar) {
            case 'n': stringLiteral += '\n'; break;
            case 't': stringLiteral += '\t'; break;
            case 'r': stringLiteral += '\r'; break;
            case '\\': stringLiteral += '\\'; break;
            case '"': stringLiteral += '"'; break;
            default: stringLiteral += static_cast<char>(lastChar); break;
          }
        } else {
          stringLiteral += static_cast<char>(lastChar);
        }
        lastChar = getNextChar();
      }

      if (lastChar == EOF)
        return tok_error; // Unterminated string

      lastChar = getNextChar(); // Skip closing quote
      return tok_string_literal;
    }

    // Integer or Float: [0-9]+ or [0-9]+.[0-9]+
    if (isdigit(lastChar)) {
      std::string numStr;
      bool isFloat = false;

      // Read digits before potential decimal point
      do {
        numStr += static_cast<char>(lastChar);
        lastChar = getNextChar();
      } while (isdigit(lastChar));

      // Check for decimal point
      if (lastChar == '.') {
        isFloat = true;
        numStr += '.';
        lastChar = getNextChar();

        // Read digits after decimal point
        while (isdigit(lastChar)) {
          numStr += static_cast<char>(lastChar);
          lastChar = getNextChar();
        }
      }

      if (isFloat) {
        floatLiteral = strtod(numStr.c_str(), nullptr);
        return tok_float_literal;
      } else {
        integerLiteral = strtoll(numStr.c_str(), nullptr, 10);
        return tok_integer_literal;
      }
    }

    // Unknown character - advance to avoid infinite loop
    lastChar = getNextChar();
    return tok_error;
  }

  /// The current token in the stream
  Token curTok = tok_eof;

  /// Location information for the current token
  LexLocation lastLocation;

  /// Identifier value when token is an identifier
  std::string identifierStr;

  /// String literal value when token is a string
  std::string stringLiteral;

  /// Integer literal value when token is an integer
  int64_t integerLiteral = 0;

  /// Float literal value when token is a float
  double floatLiteral = 0.0;

  /// The last character read from the input
  int lastChar = ' ';

  /// Current line number in the input
  int curLineNum = 0;

  /// Current column number in the input
  int curCol = 0;

  /// Buffer for the current line of input
  llvm::StringRef curLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n')
      ++current;
    if (current <= end && *current)
      ++current;
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    DBGS("got line: " << result);
    return result;
  }
  const char *current, *end;
};

} // inline namespace CamlParse
} // namespace ocamlc2

#undef DEBUG_TYPE
