#include "ocamlc2/CamlParse/Parse.h"
#include "ocamlc2/CamlParse/AST.h"
#include "ocamlc2/CamlParse/Lex.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <vector>
#include <memory>

#define DEBUG_TYPE "camlparse"
#include "ocamlc2/Support/Debug.h.inc"

#define FAIL(...)                                                              \
  DBGS(__VA_ARGS__);                                                           \
  return nullptr;

#define ORFAIL(expr, ...)                                                      \
  if (!(expr)) {                                                               \
    FAIL(__VA_ARGS__);                                                         \
  }

namespace ocamlc2 {
inline namespace CamlParse {

class Parser {
public:
  Parser(Lexer &lexer) : lexer(lexer) {
    // Prime the first token
    getNextToken();
  }

  // Main entry point for parsing a compilation unit
  std::unique_ptr<CompilationUnitAST> parseCompilationUnit() {
    DBGS("Parsing compilation unit\n");
    auto loc = createLocation();
    std::vector<std::unique_ptr<StructureAST>> structures;

    // The parse tree output should start with a list of structure items
    ORFAIL(curTok == Token::tok_bracket_open,
           "Expected '[' at the start of compilation unit, got token " 
           << getTokenName(curTok) << "\n");
    getNextToken(); // Consume '['

    // Parse the list of structure items
    while (curTok != Token::tok_bracket_close && curTok != Token::tok_eof) {
      auto structure = parseStructure();
      ORFAIL(structure,
             "Failed to parse structure in compilation unit\n");
      structures.push_back(std::move(structure));
      
      // Structures might be separated by commas
      if (curTok == Token::tok_comma) {
        getNextToken(); // Consume ','
      }
    }

    ORFAIL(curTok == Token::tok_bracket_close,
           "Expected ']' at the end of compilation unit, got token " 
           << getTokenName(curTok) << "\n");
    getNextToken(); // Consume ']'

    return std::make_unique<CompilationUnitAST>(loc, std::move(structures));
  }

private:
  Lexer &lexer;
  Token curTok;

  // Helper to advance to the next token
  Token getNextToken() {
    Token newToken = lexer.getNextToken();
    DBGS("Next token: " << getTokenName(newToken) << " at line " << lexer.getLine() 
          << ", col " << lexer.getCol() << "\n");
    return curTok = newToken;
  }

  // Create a Location object from the current lexer position
  Location createLocation() {
    auto lexLoc = lexer.getLastLocation();
    return Location(lexLoc.startLine, lexLoc.startCol, lexLoc.endLine, lexLoc.endCol);
  }

  // Parse a structure (module) definition
  std::unique_ptr<StructureAST> parseStructure() {
    DBGS("Parsing structure\n");
    auto loc = createLocation();
    std::vector<std::unique_ptr<StructureItemAST>> items;

    // Parse structure_item
    if (curTok == Token::tok_structure_item) {
      getNextToken(); // Consume 'structure_item'
      
      // Parse location information if present (test/t1.ml[2,29+0]..[2,29+10])
      if (parseSourceLocation()) {
        // Parsing of source location successful
      }

      // Parse items in the structure
      auto item = parseStructureItem();
      ORFAIL(item,
             "Failed to parse structure item in structure definition\n");
      items.push_back(std::move(item));
    } else {
      ORFAIL(curTok == Token::tok_structure_item,
             "Expected 'structure_item' at the start of a structure definition, got token " 
             << getTokenName(curTok) << "\n");
    }

    return std::make_unique<StructureAST>(loc, std::move(items));
  }

  // Parse a structure item (top-level declaration)
  std::unique_ptr<StructureItemAST> parseStructureItem() {
    DBGS("Parsing structure item\n");
    auto loc = createLocation();

    // Check for specific structure item types
    if (curTok == Token::tok_pstr_value) {
      getNextToken(); // Consume 'Pstr_value'

      // Parse Rec/Nonrec flag
      RecFlag recFlag = RecFlag::Nonrecursive;
      if (curTok == Token::tok_rec) {
        recFlag = RecFlag::Recursive;
        getNextToken(); // Consume 'Rec'
      } else if (curTok == Token::tok_nonrec) {
        recFlag = RecFlag::Nonrecursive;
        getNextToken(); // Consume 'Nonrec'
      }

      // Parse list of value definitions
      std::vector<std::unique_ptr<ValueDefinitionAST>> definitions;
      
      // Expect a list of definitions in brackets
      ORFAIL(curTok == Token::tok_bracket_open,
             "Expected '[' before value definitions, got token " 
             << getTokenName(curTok) << "\n");
      getNextToken(); // Consume '['

      while (curTok != Token::tok_bracket_close && curTok != Token::tok_eof) {
        // Debug information to see current token
        DBGS("Current token in value definition list: " << getTokenName(curTok) << "\n");
        
        // Each definition could start with a <def> marker (which may be tokenized as angle brackets)
        // or directly contain pattern and expression
        if (curTok == Token::tok_def) {
          getNextToken(); // Consume 'def'
          
          auto valueDef = parseValueDefinition(recFlag);
          ORFAIL(valueDef,
                 "Failed to parse value definition in structure item\n");
          definitions.push_back(std::move(valueDef));
        } else if (curTok == Token::tok_angle_open) {
          // Handle <def> token if it's tokenized as < def >
          getNextToken(); // Consume '<'
          
          if (curTok == Token::tok_identifier && lexer.getIdentifier() == "def") {
            getNextToken(); // Consume 'def'
            
            if (curTok == Token::tok_angle_close) {
              getNextToken(); // Consume '>'
            }
            
            auto valueDef = parseValueDefinition(recFlag);
            ORFAIL(valueDef,
                   "Failed to parse value definition in structure item\n");
            definitions.push_back(std::move(valueDef));
          } else {
            FAIL("Expected 'def' after '<' for value definition, got token " 
                 << getTokenName(curTok) << "\n");
            return nullptr;
          }
        } else if (curTok == Token::tok_pattern) {
          // Some versions might have pattern directly without <def>
          auto valueDef = parseValueDefinition(recFlag);
          ORFAIL(valueDef,
                 "Failed to parse value definition in structure item\n");
          definitions.push_back(std::move(valueDef));
        } else {
          FAIL("Expected '<def>' or 'pattern' for value definition, got token " 
               << getTokenName(curTok));
        }
        
        // Definitions might be separated by commas
        if (curTok == Token::tok_comma) {
          getNextToken(); // Consume ','
        }
      }

      ORFAIL(curTok == Token::tok_bracket_close,
             "Expected ']' after value definitions, got token " 
             << getTokenName(curTok) << "\n");
      getNextToken(); // Consume ']'

      return std::make_unique<StructureValueAST>(loc, recFlag, std::move(definitions));
    } else if (curTok == Token::tok_identifier) {
      // Handle other structure item types as needed
      std::string itemType = lexer.getIdentifier().str();
      DBGS("Found structure item of type: " << itemType << "\n");
      getNextToken();
      return std::make_unique<StructureItemAST>(loc, StructureItemAST::Str_Value);
    }

    FAIL("Unknown structure item type, got token " 
         << getTokenName(curTok) << "\n");
  }

  // Parse a value definition
  std::unique_ptr<ValueDefinitionAST> parseValueDefinition(RecFlag recFlag) {
    DBGS("Parsing value definition\n");
    auto loc = createLocation();

    // Parse pattern
    auto pattern = parsePattern();
    ORFAIL(pattern,
           "Failed to parse pattern in value definition\n");

    // Parse expression
    auto expression = parseExpression();
    ORFAIL(expression,
           "Failed to parse expression in value definition\n");

    return std::make_unique<ValueDefinitionAST>(loc, recFlag, std::move(pattern), std::move(expression));
  }

  // Parse a pattern
  std::unique_ptr<PatternAST> parsePattern() {
    DBGS("Parsing pattern\n");
    auto loc = createLocation();

    if (curTok == Token::tok_pattern) {
      getNextToken(); // Consume 'pattern'
      
      // Parse location information if present
      if (parseSourceLocation()) {
        // Parsing of source location successful
      }

      // Look for specific pattern types
      if (curTok == Token::tok_ppat_var) {
        getNextToken(); // Consume 'Ppat_var'
        
        // Expect an identifier for variable name
        ORFAIL(curTok == Token::tok_string_literal || curTok == Token::tok_identifier,
               "Expected identifier name for Ppat_var, got token " 
               << getTokenName(curTok) << "\n");
        
        std::string name;
        if (curTok == Token::tok_string_literal) {
          name = lexer.getStringLiteral().str();
        } else {
          name = lexer.getIdentifier().str();
        }
        getNextToken(); // Consume identifier
        
        return std::make_unique<PatternVariableAST>(loc, std::move(name));
      } else if (curTok == Token::tok_ppat_constant) {
        getNextToken(); // Consume 'Ppat_constant'
        
        // Parse the constant expression
        auto constant = parseConstant();
        ORFAIL(constant,
               "Failed to parse constant in pattern\n");
        
        return std::make_unique<PatternConstantAST>(loc, std::move(constant));
      } else if (curTok == Token::tok_identifier) {
        // Handle other pattern types as needed
        std::string patternType = lexer.getIdentifier().str();
        DBGS("Found pattern of type: " << patternType << "\n");
        getNextToken(); // Consume pattern type
        
        if (patternType == "Ppat_any") {
          return std::make_unique<PatternAnyAST>(loc);
        } else if (patternType == "Ppat_construct") {
          // Parse constructor name
          ORFAIL(curTok == Token::tok_string_literal || curTok == Token::tok_identifier,
                 "Expected constructor name, got token " 
                 << getTokenName(curTok) << "\n");
          
          std::string constructor;
          if (curTok == Token::tok_string_literal) {
            constructor = lexer.getStringLiteral().str();
          } else {
            constructor = lexer.getIdentifier().str();
          }
          getNextToken(); // Consume constructor name
          
          // Check if there's an argument
          std::optional<std::unique_ptr<PatternAST>> argument;
          if (curTok == Token::tok_some) {
            getNextToken(); // Consume 'Some'
            auto arg = parsePattern();
            ORFAIL(arg,
                   "Failed to parse argument in pattern\n");
            argument = std::move(arg);
          } else if (curTok == Token::tok_none) {
            getNextToken(); // Consume 'None'
            // No argument
          }
          
          return std::make_unique<PatternConstructAST>(loc, std::move(constructor), std::move(argument));
        } else {
          FAIL("Unsupported pattern type: " << patternType << "\n");
        }
      } else {
        FAIL("Expected pattern type (Ppat_var, Ppat_constant, etc.), got token " 
             << getTokenName(curTok) << "\n");
      }
    } else {
      FAIL("Expected 'pattern' token, got token " 
           << getTokenName(curTok) << "\n");
    }
    FAIL("Unknown pattern type\n");
  }

  // Parse an expression
  std::unique_ptr<ExpressionAST> parseExpression() {
    DBGS("Parsing expression\n");
    auto loc = createLocation();
    
    if (curTok == Token::tok_expression) {
      getNextToken(); // Consume 'expression'
      
      // Parse location information if present
      if (parseSourceLocation()) {
        // Parsing of source location successful
      }

      // Look for specific expression types
      if (curTok == Token::tok_pexp_constant) {
        getNextToken(); // Consume 'Pexp_constant'
        
        // Parse the constant
        auto constant = parseConstant();
        ORFAIL(constant,
               "Failed to parse constant in expression\n");
        
        return std::make_unique<ExpressionConstantAST>(loc, std::move(constant));
      } else if (curTok == Token::tok_pexp_ident) {
        getNextToken(); // Consume 'Pexp_ident'
        
        // Expect an identifier
        ORFAIL(curTok == Token::tok_string_literal || curTok == Token::tok_identifier,
               "Expected identifier for Pexp_ident, got token " 
               << getTokenName(curTok) << "\n");
        
        std::string name;
        if (curTok == Token::tok_string_literal) {
          name = lexer.getStringLiteral().str();
        } else {
          name = lexer.getIdentifier().str();
        }
        getNextToken(); // Consume identifier
        
        return std::make_unique<ExpressionVariableAST>(loc, std::move(name));
      } else if (curTok == Token::tok_pexp_apply) {
        getNextToken(); // Consume 'Pexp_apply'
        
        // Parse the function being applied
        auto function = parseExpression();
        ORFAIL(function,
               "Failed to parse function in expression\n");
        
        // Parse arguments
        std::vector<std::pair<ArgLabel, std::unique_ptr<ExpressionAST>>> arguments;
        
        // Expect a list of arguments in brackets
        ORFAIL(curTok == Token::tok_bracket_open,
               "Expected '[' before arguments, got token " 
               << getTokenName(curTok) << "\n");
        getNextToken(); // Consume '['
        
        while (curTok != Token::tok_bracket_close && curTok != Token::tok_eof) {
          // Each argument should start with <arg>
          if (curTok == Token::tok_arg) {
            getNextToken(); // Consume 'arg'
            
            // Parse label
            ArgLabel label = ArgLabel::Nolabel;
            if (curTok == Token::tok_nolabel) {
              label = ArgLabel::Nolabel;
              getNextToken(); // Consume 'Nolabel'
            } else if (curTok == Token::tok_identifier) {
              std::string labelType = lexer.getIdentifier().str();
              if (labelType == "Labelled") {
                label = ArgLabel::Labelled;
                getNextToken(); // Consume 'Labelled'
              } else if (labelType == "Optional") {
                label = ArgLabel::Optional;
                getNextToken(); // Consume 'Optional'
              }
            }
            
            // Parse expression for the argument
            auto argExpr = parseExpression();
            ORFAIL(argExpr,
                   "Failed to parse argument expression\n");
            
            arguments.push_back(std::make_pair(label, std::move(argExpr)));
          } else {
            FAIL("Expected 'arg' for function application argument, got token " 
                 << getTokenName(curTok) << "\n");
          }
          
          // Arguments might be separated by commas
          if (curTok == Token::tok_comma) {
            getNextToken(); // Consume ','
          }
        }
        
        ORFAIL(curTok == Token::tok_bracket_close,
               "Expected ']' after arguments, got token " 
               << getTokenName(curTok) << "\n");
        getNextToken(); // Consume ']'
        
        return std::make_unique<ExpressionApplyAST>(loc, std::move(function), std::move(arguments));
      } else if (curTok == Token::tok_pexp_function) {
        getNextToken(); // Consume 'Pexp_function'
        
        // Parse parameters
        std::vector<std::unique_ptr<ParameterAST>> parameters;
        
        // Expect a list of parameters in brackets
        ORFAIL(curTok == Token::tok_bracket_open,
               "Expected '[' before parameters, got token " 
               << getTokenName(curTok) << "\n");
        getNextToken(); // Consume '['
        
        while (curTok != Token::tok_bracket_close && curTok != Token::tok_eof) {
          if (curTok == Token::tok_pparam_val) {
            getNextToken(); // Consume 'Pparam_val'
            
            // Parse source location if present
            if (parseSourceLocation()) {
              // Parsing of source location successful
            }
            
            // Parse label
            ArgLabel label = ArgLabel::Nolabel;
            std::string labelName = "";
            if (curTok == Token::tok_nolabel) {
              label = ArgLabel::Nolabel;
              getNextToken(); // Consume 'Nolabel'
            } else if (curTok == Token::tok_identifier) {
              std::string labelType = lexer.getIdentifier().str();
              if (labelType == "Labelled") {
                label = ArgLabel::Labelled;
                getNextToken(); // Consume 'Labelled'
                
                // Get the label name
                if (curTok == Token::tok_string_literal) {
                  labelName = lexer.getStringLiteral().str();
                  getNextToken(); // Consume label name
                }
              } else if (labelType == "Optional") {
                label = ArgLabel::Optional;
                getNextToken(); // Consume 'Optional'
                
                // Get the label name
                if (curTok == Token::tok_string_literal) {
                  labelName = lexer.getStringLiteral().str();
                  getNextToken(); // Consume label name
                }
              }
            }
            
            // Parse default value for the parameter (if any)
            std::optional<std::unique_ptr<ASTNode>> defaultValue;
            if (curTok == Token::tok_some) {
              getNextToken(); // Consume 'Some'
              // TODO: Parse default value expression
            } else if (curTok == Token::tok_none) {
              getNextToken(); // Consume 'None'
              // No default value
            }
            
            // Parse the pattern for the parameter
            auto pattern = parsePattern();
            ORFAIL(pattern,
                   "Failed to parse pattern for function parameter\n");
            
            parameters.push_back(std::make_unique<ParameterAST>(loc, label, labelName, std::move(pattern), std::move(defaultValue)));
          } else {
            FAIL("Expected 'Pparam_val' for function parameter, got token " 
                 << getTokenName(curTok) << "\n");
          }
          
          // Parameters might be separated by commas
          if (curTok == Token::tok_comma) {
            getNextToken(); // Consume ','
          }
        }
        
        ORFAIL(curTok == Token::tok_bracket_close,
               "Expected ']' after parameters, got token " 
               << getTokenName(curTok) << "\n");
        getNextToken(); // Consume ']'
        
        // Parse return type (if any)
        std::optional<std::unique_ptr<ASTNode>> returnType;
        if (curTok == Token::tok_some) {
          getNextToken(); // Consume 'Some'
          // TODO: Parse return type
        } else if (curTok == Token::tok_none) {
          getNextToken(); // Consume 'None'
          // No return type
        }
        
        // Parse function body
        if (curTok == Token::tok_pfunction_body) {
          getNextToken(); // Consume 'Pfunction_body'
        }
        
        auto body = parseExpression();
        ORFAIL(body,
               "Failed to parse function body\n");
        
        return std::make_unique<ExpressionFunctionAST>(loc, std::move(parameters), std::move(body), std::move(returnType));
      } else if (curTok == Token::tok_identifier) {
        // Handle other expression types as needed
        std::string exprType = lexer.getIdentifier().str();
        DBGS("Found expression of type: " << exprType << "\n");
        getNextToken(); // Consume expression type
        
        // Implementation for other expression types would go here
      } else {
        FAIL("Unexpected token in expression: " << getTokenName(curTok) << "\n");
      }
    } else {
      FAIL("Expected 'expression' token, got " << getTokenName(curTok) << "\n");
    }
    
    FAIL("Falling back to dummy expression\n");
  }

  // Parse a constant
  std::unique_ptr<ConstantAST> parseConstant() {
    DBGS("Parsing constant\n");
    auto loc = createLocation();
    
    if (curTok == Token::tok_pconst_int) {
      getNextToken(); // Consume 'PConst_int'
      
      // Parse the integer value
      ORFAIL(curTok == Token::tok_integer_literal || curTok == Token::tok_parenthese_open,
             "Expected integer value for PConst_int, got token " 
             << getTokenName(curTok) << "\n");
      
      int64_t value;
      if (curTok == Token::tok_integer_literal) {
        value = lexer.getIntegerLiteral();
        getNextToken(); // Consume integer
      } else {
        // Handle parenthesized value
        getNextToken(); // Consume '('
        ORFAIL(curTok == Token::tok_integer_literal,
               "Expected integer value in parentheses, got token " 
               << getTokenName(curTok) << "\n");
        value = lexer.getIntegerLiteral();
        getNextToken(); // Consume integer
        
        // Skip optional suffix information
        if (curTok == Token::tok_comma) {
          getNextToken(); // Consume ','
          // Skip suffix token
          if (curTok == Token::tok_none) {
            getNextToken(); // Consume 'None'
          } else if (curTok == Token::tok_some) {
            getNextToken(); // Consume 'Some'
            if (curTok == Token::tok_string_literal) {
              getNextToken(); // Consume string
            }
          }
        }
        
        ORFAIL(curTok == Token::tok_parenthese_close,
               "Expected ')' after integer value, got token " 
               << getTokenName(curTok) << "\n");
        getNextToken(); // Consume ')'
      }
      
      return std::make_unique<IntConstantAST>(loc, value);
    } else if (curTok == Token::tok_pconst_string) {
      getNextToken(); // Consume 'PConst_string'
      
      // Parse the string value (possibly with location and delimiter)
      ORFAIL(curTok == Token::tok_string_literal || curTok == Token::tok_parenthese_open,
             "Expected string value for PConst_string, got token " 
             << getTokenName(curTok) << "\n");
      
      
      std::string value;
      if (curTok == Token::tok_string_literal) {
        value = lexer.getStringLiteral().str();
        getNextToken(); // Consume string
      } else {
        // Handle parenthesized value with location and delimiter
        getNextToken(); // Consume '('
        ORFAIL(curTok == Token::tok_string_literal,
               "Expected string value in parentheses, got token " 
               << getTokenName(curTok) << "\n");
        value = lexer.getStringLiteral().str();
        getNextToken(); // Consume string
        
        // Skip location and delimiter information
        if (curTok == Token::tok_comma) {
          getNextToken(); // Consume ','
          // Skip location token
          parseSourceLocation();
          
          if (curTok == Token::tok_comma) {
            getNextToken(); // Consume ','
            // Skip delimiter token
            if (curTok == Token::tok_none) {
              getNextToken(); // Consume 'None'
            } else if (curTok == Token::tok_some) {
              getNextToken(); // Consume 'Some'
              if (curTok == Token::tok_string_literal) {
                getNextToken(); // Consume string
              }
            }
          }
        }
        
        ORFAIL(curTok == Token::tok_parenthese_close,
               "Expected ')' after string value, got token " 
               << getTokenName(curTok) << "\n");
        getNextToken(); // Consume ')'
      }
      
      return std::make_unique<StringConstantAST>(loc, std::move(value));
    } else if (curTok == Token::tok_pconst_float) {
      getNextToken(); // Consume 'PConst_float'
      
      // Parse the float value
      ORFAIL(curTok == Token::tok_float_literal || curTok == Token::tok_string_literal,
             "Expected float value for PConst_float, got token " 
             << getTokenName(curTok) << "\n");
      
      
      double value;
      if (curTok == Token::tok_float_literal) {
        value = lexer.getFloatLiteral();
        getNextToken(); // Consume float
      } else {
        // In OCaml AST output, floats are often represented as strings
        std::string floatStr = lexer.getStringLiteral().str();
        value = std::stod(floatStr);
        getNextToken(); // Consume string
      }
      
      return std::make_unique<FloatConstantAST>(loc, value);
    } else {
      FAIL("Unknown constant type, got token " 
           << getTokenName(curTok) << "\n");
    }
  }

  // Parse source location information (test/t1.ml[2,29+0]..[2,29+10])
  // Returns true if location was parsed successfully
  bool parseSourceLocation() {
    DBGS("Parsing source location\n");
    // Skip location information if present
    if (curTok == Token::tok_parenthese_open) {
      int depth = 1;
      getNextToken(); // Consume '('
      
      // Skip everything until matching closing parenthesis
      while (depth > 0 && curTok != Token::tok_eof) {
        if (curTok == Token::tok_parenthese_open) {
          depth++;
        } else if (curTok == Token::tok_parenthese_close) {
          depth--;
        }
        
        if (depth > 0) {
          getNextToken();
        }
      }
      
      if (curTok == Token::tok_parenthese_close) {
        getNextToken(); // Consume ')'
      }
      
      return true;
    }
    
    return false;
  }
};

std::unique_ptr<CompilationUnitAST> parse(std::istream &file, llvm::StringRef filepath) {
  DBGS("Reading file content\n");
  // Read the entire file content into a string
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  
  DBGS("Creating lexer\n");
  // Create a lexer for the file content
  LexerBuffer lexer(content.c_str(), content.c_str() + content.size(), 
                    filepath.str());
  
  DBGS("Starting parser\n");
  // Create a parser and parse the compilation unit
  Parser parser(lexer);
  return parser.parseCompilationUnit();
}

} // inline namespace CamlParse
} // namespace ocamlc2
