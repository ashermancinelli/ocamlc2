#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/Colors.h"
#include "ocamlc2/Parse/TSAdaptor.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Location.h>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#define DEBUG_TYPE "ast"
#include "ocamlc2/Support/Debug.h.inc"

using namespace ocamlc2;
namespace fs = std::filesystem;

llvm::StringRef ASTNode::getName(const ASTNode &node) {
  return getName(node.getKind());
}

llvm::StringRef ASTNode::getName(ASTNodeKind kind) {
  switch (kind) {
    case Node_Number: return "Number";
    case Node_ValuePath: return "ValuePath";
    case Node_ConstructorPath: return "ConstructorPath";
    case Node_TypeConstructorPath: return "TypeConstructorPath";
    case Node_Application: return "Application";
    case Node_InfixExpression: return "InfixExpression";
    case Node_ParenthesizedExpression: return "ParenthesizedExpression";
    case Node_MatchExpression: return "MatchExpression";
    case Node_ForExpression: return "ForExpression";
    case Node_LetExpression: return "LetExpression";
    case Node_MatchCase: return "MatchCase";
    case Node_LetBinding: return "LetBinding";
    case Node_CompilationUnit: return "CompilationUnit";
    case Node_ValuePattern: return "ValuePattern";
    case Node_ConstructorPattern: return "ConstructorPattern";
    case Node_TypedPattern: return "TypedPattern";
    case Node_TypeDefinition: return "TypeDefinition";
    case Node_ValueDefinition: return "ValueDefinition";
    case Node_ExpressionItem: return "ExpressionItem";
    case Node_TypeBinding: return "TypeBinding";
    case Node_VariantDeclaration: return "VariantDeclaration";
    case Node_ConstructorDeclaration: return "ConstructorDeclaration";
  }
}

mlir::Location ASTNode::getMLIRLocation(mlir::MLIRContext &context) const {
  return mlir::FileLineColLoc::get(&context, loc().filename, loc().line,
                                   loc().column);
}

// Debugging functions
void dumpTSNode(TSNode node, const TSTreeAdaptor &adaptor, int indent = 0) {
  std::string indentation(indent * 2, ' ');
  const char* nodeType = ts_node_type(node);
  uint32_t start = ts_node_start_byte(node);
  uint32_t end = ts_node_end_byte(node);
  std::string text = adaptor.getSource().substr(start, end - start).str();
  
  if (text.find('\n') != std::string::npos) {
    text = "<multiline>";
  }

  std::cerr << indentation << ANSIColors::cyan << nodeType << ANSIColors::reset
            << ": " << ANSIColors::italic << text << ANSIColors::reset
            << std::endl;

  uint32_t childCount = ts_node_child_count(node);
  for (uint32_t i = 0; i < childCount; ++i) {
    TSNode child = ts_node_child(node, i);
    dumpTSNode(child, adaptor, indent + 1);
  }
}

// Forward declarations of conversion functions
std::unique_ptr<ASTNode> convertNode(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<NumberExprAST> convertNumber(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ValuePathAST> convertValuePath(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ConstructorPathAST> convertConstructorPath(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypeConstructorPathAST> convertTypeConstructorPath(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ApplicationExprAST> convertApplicationExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<InfixExpressionAST> convertInfixExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ParenthesizedExpressionAST> convertParenthesizedExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<MatchExpressionAST> convertMatchExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<MatchCaseAST> convertMatchCase(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ForExpressionAST> convertForExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<LetExpressionAST> convertLetExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ValuePatternAST> convertValuePattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ConstructorPatternAST> convertConstructorPattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypedPatternAST> convertTypedPattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypeDefinitionAST> convertTypeDefinition(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ValueDefinitionAST> convertValueDefinition(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ExpressionItemAST> convertExpressionItem(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypeBindingAST> convertTypeBinding(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<VariantDeclarationAST> convertVariantDeclaration(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ConstructorDeclarationAST> convertConstructorDeclaration(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<LetBindingAST> convertLetBinding(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<CompilationUnitAST> convertCompilationUnit(TSNode node, const TSTreeAdaptor &adaptor);

// Set of known/supported node types for better error reporting
std::unordered_set<std::string> knownNodeTypes = {
    "compilation_unit", "number", "value_path", "constructor_path",
    "type_constructor_path", "application_expression", "infix_expression",
    "parenthesized_expression", "match_expression", "match_case",
    "value_pattern", "constructor_pattern", "typed_pattern", "type_definition",
    "value_definition", "expression_item", "type_binding", "variant_declaration",
    "constructor_declaration", "let_binding", "value_name", "type_constructor",
    "constructor_name", "add_operator", "subtract_operator", "multiply_operator",
    "division_operator", "concat_operator", "and_operator", "or_operator",
    "equal_operator", "parameter", "field_get_expression", "for_expression", 
    "let_expression", "in",
    ";", ";;", "(", ")", ":", "=", "->", "|", "of", "with", "match", "type", "let",
    "do", "done", "to", "downto"
};

// Helper functions
Location getLocation(TSNode node, const TSTreeAdaptor &adaptor) {
  Location loc;
  TSPoint startPoint = ts_node_start_point(node);
  loc.filename = adaptor.getFilename().str();
  loc.line = startPoint.row + 1; // Convert from 0-based to 1-based
  loc.column = startPoint.column + 1;
  return loc;
}

// Helper to get the actual text content of a node
std::string getNodeText(TSNode node, const TSTreeAdaptor &adaptor) {
  uint32_t start = ts_node_start_byte(node);
  uint32_t end = ts_node_end_byte(node);
  return adaptor.getSource().substr(start, end - start).str();
}

// Map of operator node types to their actual operators
std::unordered_map<std::string, std::string> operatorMap = {
    {"add_operator", "+"},
    {"subtract_operator", "-"},
    {"multiply_operator", "*"},
    {"division_operator", "/"},
    {"concat_operator", "^"},
    {"and_operator", "&&"},
    {"or_operator", "||"},
    {"equal_operator", "="}
};

std::unique_ptr<ASTNode> ocamlc2::parse(const std::string &source, const std::string &filename) {
  TSTreeAdaptor tree(filename, source);
  TSNode rootNode = ts_tree_root_node(tree);
  
  // Debug the tree-sitter parse tree if debug is enabled
  DBGS("Tree-sitter parse tree:\n");
  DBG(dumpTSNode(rootNode, tree));
  
  return convertNode(rootNode, tree);
}

std::unique_ptr<ASTNode> ocamlc2::parse(const std::filesystem::path &filepath) {
  assert(fs::exists(filepath) && "File does not exist");
  std::string source = must(slurpFile(filepath.string()));
  
  return parse(source, filepath.string());
}

std::unique_ptr<ASTNode> convertNode(TSNode node, const TSTreeAdaptor &adaptor) {
  if (ts_node_is_null(node)) {
    return nullptr;
  }
  
  const char* nodeType = ts_node_type(node);
  StringRef type(nodeType);
  
  if (type == "compilation_unit")
    return convertCompilationUnit(node, adaptor);
  else if (type == "number")
    return convertNumber(node, adaptor);
  else if (type == "value_path")
    return convertValuePath(node, adaptor);
  else if (type == "constructor_path")
    return convertConstructorPath(node, adaptor);
  else if (type == "type_constructor_path")
    return convertTypeConstructorPath(node, adaptor);
  else if (type == "application_expression")
    return convertApplicationExpr(node, adaptor);
  else if (type == "infix_expression")
    return convertInfixExpr(node, adaptor);
  else if (type == "parenthesized_expression")
    return convertParenthesizedExpr(node, adaptor);
  else if (type == "match_expression")
    return convertMatchExpr(node, adaptor);
  else if (type == "match_case")
    return convertMatchCase(node, adaptor);
  else if (type == "for_expression")
    return convertForExpr(node, adaptor);
  else if (type == "let_expression")
    return convertLetExpr(node, adaptor);
  else if (type == "value_pattern")
    return convertValuePattern(node, adaptor);
  else if (type == "constructor_pattern")
    return convertConstructorPattern(node, adaptor);
  else if (type == "typed_pattern")
    return convertTypedPattern(node, adaptor);
  else if (type == "type_definition")
    return convertTypeDefinition(node, adaptor);
  else if (type == "value_definition")
    return convertValueDefinition(node, adaptor);
  else if (type == "expression_item")
    return convertExpressionItem(node, adaptor);
  else if (type == "type_binding")
    return convertTypeBinding(node, adaptor);
  else if (type == "variant_declaration")
    return convertVariantDeclaration(node, adaptor);
  else if (type == "constructor_declaration")
    return convertConstructorDeclaration(node, adaptor);
  else if (type == "let_binding")
    return convertLetBinding(node, adaptor);
  else if (type == "field_get_expression") {
    // For field_get_expression, we can treat it as a special case of value_path
    auto children = childrenNodes(node);
    std::vector<std::string> path;
    
    // Process all parts of the field path
    for (auto [childType, child] : children) {
      if (childType != "." && childType != "field_name") {
        auto expr = convertNode(child, adaptor);
        if (expr) {
          if (auto *valuePath = llvm::dyn_cast<ValuePathAST>(expr.get())) {
            // Extract the existing path and add to it
            auto existingPath = valuePath->getPath();
            path.insert(path.end(), existingPath.begin(), existingPath.end());
          } else {
            // If not a ValuePath, get the text directly
            path.push_back(getNodeText(child, adaptor));
          }
        }
      } else if (childType == "field_name") {
        path.push_back(getNodeText(child, adaptor));
      }
    }
    
    // Add field_name and field_path to known node types
    knownNodeTypes.insert("field_name");
    knownNodeTypes.insert("field_path");
    
    return std::make_unique<ValuePathAST>(getLocation(node, adaptor), std::move(path));
  }
  
  // Default: try processing children nodes if this is an unknown node type
  // This helps with handling various expression and pattern types that might not have direct handlers
  auto children = childrenNodes(node);
  if (!children.empty()) {
    for (auto [childType, child] : children) {
      // Skip tokens and try to process actual nodes
      if (childType != ";" && childType != ";;" && childType != "(" && childType != ")" &&
          childType != ":" && childType != "=" && childType != "->" && childType != "|" &&
          childType != "of" && childType != "with" && childType != "match" && 
          childType != "type" && childType != "let") {
        auto result = convertNode(child, adaptor);
        if (result) {
          return result;
        }
      }
    }
  }
  
  // Log warning for truly unsupported node types
  if (knownNodeTypes.find(std::string(nodeType)) == knownNodeTypes.end()) {
    std::cerr << "Warning: Unsupported node type: " << nodeType << std::endl;
    DBGS("Unsupported node type: " << nodeType << "\n");
    DBG(dumpTSNode(node, adaptor));
  }
  
  return nullptr;
}

std::unique_ptr<NumberExprAST> convertNumber(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "number") {
    return nullptr;
  }
  
  std::string value = getNodeText(node, adaptor);
  return std::make_unique<NumberExprAST>(getLocation(node, adaptor), value);
}

std::unique_ptr<ValuePathAST> convertValuePath(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "value_path") {
    return nullptr;
  }
  
  std::vector<std::string> path;
  auto children = childrenNodes(node);
  
  for (auto [type, child] : children) {
    if (type == "value_name") {
      path.push_back(getNodeText(child, adaptor));
    }
  }
  
  return std::make_unique<ValuePathAST>(getLocation(node, adaptor), std::move(path));
}

std::unique_ptr<ConstructorPathAST> convertConstructorPath(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "constructor_path") {
    return nullptr;
  }
  
  std::vector<std::string> path;
  auto children = childrenNodes(node);
  
  for (auto [type, child] : children) {
    if (type == "constructor_name") {
      path.push_back(getNodeText(child, adaptor));
    }
  }
  
  return std::make_unique<ConstructorPathAST>(getLocation(node, adaptor), std::move(path));
}

std::unique_ptr<TypeConstructorPathAST> convertTypeConstructorPath(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "type_constructor_path") {
    return nullptr;
  }
  
  std::vector<std::string> path;
  auto children = childrenNodes(node);
  
  for (auto [type, child] : children) {
    if (type == "type_constructor") {
      path.push_back(getNodeText(child, adaptor));
    }
  }
  
  return std::make_unique<TypeConstructorPathAST>(getLocation(node, adaptor), std::move(path));
}

std::unique_ptr<ApplicationExprAST> convertApplicationExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "application_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  if (children.empty()) {
    return nullptr;
  }
  
  // First child is the function
  auto function = convertNode(children[0].second, adaptor);
  if (!function) {
    return nullptr;
  }
  
  // Remaining children are arguments
  std::vector<std::unique_ptr<ASTNode>> arguments;
  for (size_t i = 1; i < children.size(); ++i) {
    auto arg = convertNode(children[i].second, adaptor);
    if (arg) {
      arguments.push_back(std::move(arg));
    }
  }
  
  return std::make_unique<ApplicationExprAST>(
    getLocation(node, adaptor),
    std::move(function),
    std::move(arguments)
  );
}

std::unique_ptr<InfixExpressionAST> convertInfixExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "infix_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  if (children.size() < 3) {
    return nullptr;
  }
  
  auto lhs = convertNode(children[0].second, adaptor);
  
  // Get the operator - handle different operator node types
  std::string op;
  const auto& [opType, opNode] = children[1];
  auto it = operatorMap.find(opType.str());
  if (it != operatorMap.end()) {
    op = it->second;
  } else {
    op = getNodeText(opNode, adaptor);
  }
  
  auto rhs = convertNode(children[2].second, adaptor);
  
  if (!lhs || !rhs) {
    return nullptr;
  }
  
  return std::make_unique<InfixExpressionAST>(
    getLocation(node, adaptor),
    std::move(lhs),
    op,
    std::move(rhs)
  );
}

std::unique_ptr<ParenthesizedExpressionAST> convertParenthesizedExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "parenthesized_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> expr = nullptr;
  
  // Find the expression inside the parentheses
  for (auto [type, child] : children) {
    if (type != "(" && type != ")") {
      expr = convertNode(child, adaptor);
      if (expr) break;
    }
  }
  
  if (!expr) {
    return nullptr;
  }
  
  return std::make_unique<ParenthesizedExpressionAST>(
    getLocation(node, adaptor),
    std::move(expr)
  );
}

std::unique_ptr<MatchCaseAST> convertMatchCase(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "match_case") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> pattern = nullptr;
  std::unique_ptr<ASTNode> expr = nullptr;
  
  // First child is the pattern, after -> is the expression
  bool foundArrow = false;
  for (auto [type, child] : children) {
    if (type == "->") {
      foundArrow = true;
      continue;
    }
    
    if (!foundArrow) {
      pattern = convertNode(child, adaptor);
    } else {
      expr = convertNode(child, adaptor);
      if (expr) break;
    }
  }
  
  if (!pattern || !expr) {
    return nullptr;
  }
  
  return std::make_unique<MatchCaseAST>(
    getLocation(node, adaptor),
    std::move(pattern),
    std::move(expr)
  );
}

std::unique_ptr<MatchExpressionAST> convertMatchExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "match_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> value = nullptr;
  std::vector<std::unique_ptr<MatchCaseAST>> cases;
  
  // Find the match value and cases
  bool foundWith = false;
  for (auto [type, child] : children) {
    if (type == "match") {
      continue;
    } else if (type == "with") {
      foundWith = true;
      continue;
    } else if (!foundWith) {
      value = convertNode(child, adaptor);
    } else if (type == "match_case") {
      auto matchCase = convertMatchCase(child, adaptor);
      if (matchCase) {
        cases.push_back(std::move(matchCase));
      }
    }
  }
  
  if (!value || cases.empty()) {
    return nullptr;
  }
  
  return std::make_unique<MatchExpressionAST>(
    getLocation(node, adaptor),
    std::move(value),
    std::move(cases)
  );
}

std::unique_ptr<ValuePatternAST> convertValuePattern(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "value_pattern") {
    return nullptr;
  }
  
  std::string name = getNodeText(node, adaptor);
  return std::make_unique<ValuePatternAST>(getLocation(node, adaptor), name);
}

std::unique_ptr<ConstructorPatternAST> convertConstructorPattern(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "constructor_pattern") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ConstructorPathAST> constructor = nullptr;
  std::vector<std::unique_ptr<ASTNode>> arguments;
  
  for (auto [type, child] : children) {
    if (type == "constructor_path") {
      constructor = convertConstructorPath(child, adaptor);
    } else if (type != "constructor_path") {
      // Any other node could be an argument
      auto arg = convertNode(child, adaptor);
      if (arg) {
        arguments.push_back(std::move(arg));
      }
    }
  }
  
  if (!constructor) {
    return nullptr;
  }
  
  return std::make_unique<ConstructorPatternAST>(
    getLocation(node, adaptor),
    std::move(constructor),
    std::move(arguments)
  );
}

std::unique_ptr<TypedPatternAST> convertTypedPattern(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "typed_pattern") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> pattern = nullptr;
  std::unique_ptr<TypeConstructorPathAST> type = nullptr;
  
  // Debug children for this node
  DBGS("Typed pattern children:\n");
  for (auto [childType, child] : children) {
    DBGS("  Type: " << childType.str() << "\n");
  }
  
  // First pass - try to find pattern and type
  for (auto [childType, child] : children) {
    if (childType == "value_name") {
      std::string name = getNodeText(child, adaptor);
      pattern = std::make_unique<ValuePatternAST>(getLocation(child, adaptor), name);
    } else if (childType == "type_constructor_path") {
      type = convertTypeConstructorPath(child, adaptor);
    }
  }
  
  // Second pass - for more complex patterns or if we didn't find a simple value_name
  if (!pattern) {
    for (auto [childType, child] : children) {
      if (childType != ":" && childType != "(" && childType != ")") {
        // Try to extract any other pattern node
        auto possiblePattern = convertNode(child, adaptor);
        if (possiblePattern && !pattern) {
          pattern = std::move(possiblePattern);
        }
      }
    }
  }
  
  if (!pattern || !type) {
    DBGS("Failed to parse typed_pattern:\n");
    if (!pattern) DBGS("  Missing pattern\n");
    if (!type) DBGS("  Missing type\n");
    return nullptr;
  }
  
  return std::make_unique<TypedPatternAST>(
    getLocation(node, adaptor),
    std::move(pattern),
    std::move(type)
  );
}

std::unique_ptr<ConstructorDeclarationAST> convertConstructorDeclaration(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "constructor_declaration") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::string name;
  std::vector<std::unique_ptr<TypeConstructorPathAST>> ofTypes;
  
  for (auto [type, child] : children) {
    if (type == "constructor_name") {
      name = getNodeText(child, adaptor);
    } else if (type == "of") {
      // Skip "of" keyword
    } else if (type == "*") {
      // Skip "*" operator for tuples
    } else if (type == "type_constructor_path") {
      ofTypes.push_back(convertTypeConstructorPath(child, adaptor));
    }
  }
  
  if (name.empty()) {
    return nullptr;
  }
  
  return std::make_unique<ConstructorDeclarationAST>(
    getLocation(node, adaptor),
    name,
    std::move(ofTypes)
  );
}

std::unique_ptr<VariantDeclarationAST> convertVariantDeclaration(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "variant_declaration") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<ConstructorDeclarationAST>> constructors;
  
  for (auto [type, child] : children) {
    if (type == "constructor_declaration") {
      auto constructor = convertConstructorDeclaration(child, adaptor);
      if (constructor) {
        constructors.push_back(std::move(constructor));
      }
    }
  }
  
  if (constructors.empty()) {
    return nullptr;
  }
  
  return std::make_unique<VariantDeclarationAST>(
    getLocation(node, adaptor),
    std::move(constructors)
  );
}

std::unique_ptr<TypeBindingAST> convertTypeBinding(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "type_binding") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::string name;
  std::unique_ptr<ASTNode> definition = nullptr;
  
  for (auto [type, child] : children) {
    if (type == "type_constructor") {
      name = getNodeText(child, adaptor);
    } else if (type == "=") {
      // Skip "=" token
    } else if (type == "variant_declaration") {
      definition = convertVariantDeclaration(child, adaptor);
    }
  }
  
  if (name.empty() || !definition) {
    return nullptr;
  }
  
  return std::make_unique<TypeBindingAST>(
    getLocation(node, adaptor),
    name,
    std::move(definition)
  );
}

std::unique_ptr<TypeDefinitionAST> convertTypeDefinition(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "type_definition") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<TypeBindingAST>> bindings;
  
  for (auto [type, child] : children) {
    if (type == "type_binding") {
      auto binding = convertTypeBinding(child, adaptor);
      if (binding) {
        bindings.push_back(std::move(binding));
      }
    }
  }
  
  if (bindings.empty()) {
    return nullptr;
  }
  
  return std::make_unique<TypeDefinitionAST>(
    getLocation(node, adaptor),
    std::move(bindings)
  );
}

std::unique_ptr<LetBindingAST> convertLetBinding(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "let_binding") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::string name;
  std::vector<std::unique_ptr<ASTNode>> parameters;
  std::unique_ptr<TypeConstructorPathAST> returnType = nullptr;
  std::unique_ptr<ASTNode> body = nullptr;
  
  // First try to find a typed_pattern or value_name for the binding name
  std::string patternName;
  for (auto [type, child] : children) {
    if (type == "value_name") {
      name = getNodeText(child, adaptor);
      break;
    } else if (type == "typed_pattern") {
      // Extract name from the typed pattern
      auto patternChildren = childrenNodes(child);
      for (auto [patternType, patternChild] : patternChildren) {
        if (patternType == "value_name") {
          name = getNodeText(patternChild, adaptor);
          break;
        }
      }
      
      // If we found the name, no need to check for return type
      // as it's already included in the typed pattern
      if (!name.empty()) {
        break;
      }
    }
  }
  
  // Then process the rest of the binding
  for (auto [type, child] : children) {
    if (type == "parameter") {
      auto paramChildren = childrenNodes(child);
      bool foundParam = false;
      for (auto [paramType, paramChild] : paramChildren) {
        if (paramType == "typed_pattern") {
          auto param = convertTypedPattern(paramChild, adaptor);
          if (param) {
            parameters.push_back(std::move(param));
            foundParam = true;
          }
        }
      }
      
      // If we didn't find a typed pattern, try to convert the parameter directly
      if (!foundParam) {
        auto param = convertNode(child, adaptor);
        if (param) {
          parameters.push_back(std::move(param));
        }
      }
    } else if (type == ":") {
      // Skip ":" token
    } else if (type == "type_constructor_path") {
      returnType = convertTypeConstructorPath(child, adaptor);
    } else if (type == "=") {
      // Skip "=" token
    } else if (type != "value_name" && type != ":" && type != "=" && type != "typed_pattern") {
      // This could be the function body (various expression types)
      auto possibleBody = convertNode(child, adaptor);
      if (possibleBody) {
        body = std::move(possibleBody);
      }
    }
  }
  
  if (name.empty() || !body) {
    DBGS("Failed to parse let_binding:\n");
    if (name.empty()) DBGS("  Missing name\n");
    if (!body) DBGS("  Missing body\n");
    return nullptr;
  }
  
  return std::make_unique<LetBindingAST>(
    getLocation(node, adaptor),
    name,
    std::move(parameters),
    std::move(returnType),
    std::move(body)
  );
}

std::unique_ptr<ValueDefinitionAST> convertValueDefinition(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "value_definition") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<LetBindingAST>> bindings;
  
  for (auto [type, child] : children) {
    if (type == "let_binding") {
      auto binding = convertLetBinding(child, adaptor);
      if (binding) {
        bindings.push_back(std::move(binding));
      }
    }
  }
  
  if (bindings.empty()) {
    return nullptr;
  }
  
  return std::make_unique<ValueDefinitionAST>(
    getLocation(node, adaptor),
    std::move(bindings)
  );
}

std::unique_ptr<ExpressionItemAST> convertExpressionItem(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "expression_item") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> expression = nullptr;
  
  for (auto [type, child] : children) {
    if (type != ";;" && type != ";") {
      expression = convertNode(child, adaptor);
      if (expression) break;
    }
  }
  
  if (!expression) {
    return nullptr;
  }
  
  return std::make_unique<ExpressionItemAST>(
    getLocation(node, adaptor),
    std::move(expression)
  );
}

std::unique_ptr<CompilationUnitAST> convertCompilationUnit(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "compilation_unit") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<ASTNode>> items;
  
  for (auto [type, child] : children) {
    if (type == "type_definition" || 
        type == "value_definition" || 
        type == "expression_item") {
      auto item = convertNode(child, adaptor);
      if (item) {
        items.push_back(std::move(item));
      }
    }
  }
  
  return std::make_unique<CompilationUnitAST>(
    getLocation(node, adaptor),
    std::move(items)
  );
}

std::unique_ptr<ForExpressionAST> convertForExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "for_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::string loopVar;
  std::unique_ptr<ASTNode> startExpr = nullptr;
  std::unique_ptr<ASTNode> endExpr = nullptr;
  std::unique_ptr<ASTNode> body = nullptr;
  bool isDownto = false;
  
  for (auto [type, child] : children) {
    if (type == "for") {
      continue; // Skip "for" keyword
    } else if (type == "value_pattern") {
      // Loop variable (i)
      loopVar = getNodeText(child, adaptor);
    } else if (type == "=") {
      continue; // Skip "=" token
    } else if (type == "to" || type == "downto") {
      isDownto = (type == "downto");
      continue;
    } else if ((type == "number" || type == "value_path") && !startExpr) {
      // Start expression - can be a number or variable
      startExpr = convertNode(child, adaptor);
    } else if ((type == "number" || type == "value_path") && startExpr && !endExpr) {
      // End expression - can be a number or variable
      endExpr = convertNode(child, adaptor);
    } else if (type == "do_clause") {
      // Body is inside do_clause
      auto doChildren = childrenNodes(child);
      for (auto [doType, doChild] : doChildren) {
        if (doType != "do" && doType != "done") {
          body = convertNode(doChild, adaptor);
          break;
        }
      }
    }
  }
  
  if (loopVar.empty() || !startExpr || !endExpr || !body) {
    // Log what we found for debugging
    DBGS("Failed to parse for_expression: \n");
    if (loopVar.empty()) DBGS("  Missing loop variable\n");
    if (!startExpr) DBGS("  Missing start expression\n");
    if (!endExpr) DBGS("  Missing end expression\n");
    if (!body) DBGS("  Missing body\n");
    return nullptr;
  }
  
  return std::make_unique<ForExpressionAST>(
    getLocation(node, adaptor),
    std::move(loopVar),
    std::move(startExpr),
    std::move(endExpr),
    std::move(body),
    isDownto
  );
}

std::unique_ptr<LetExpressionAST> convertLetExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "let_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> binding = nullptr;
  std::unique_ptr<ASTNode> body = nullptr;
  bool foundIn = false;
  
  // Dump the children structure for debugging
  DBGS("Let expression children:\n");
  for (auto [type, child] : children) {
    DBGS("  Type: " << type.str() << "\n");
  }
  
  // Process let_binding directly if available, or through value_definition otherwise
  for (auto [type, child] : children) {
    if (type == "let_binding") {
      // Direct let_binding
      binding = convertLetBinding(child, adaptor);
    } else if (type == "value_definition") {
      // Value definition containing let_binding
      binding = convertValueDefinition(child, adaptor);
    } else if (type == "in") {
      foundIn = true;
      // Skip the 'in' token
      continue;
    } else if (foundIn && !body) {
      // This should be the body (anything after 'in')
      body = convertNode(child, adaptor);
    }
  }
  
  // If we didn't find a binding yet, look deeper into the children
  if (!binding) {
    for (auto [type, child] : children) {
      if (type != "in" && !foundIn) {
        // Try to find a binding in any non-in token before 'in'
        auto possibleBinding = convertNode(child, adaptor);
        if (possibleBinding) {
          binding = std::move(possibleBinding);
        }
      }
    }
  }
  
  if (!binding || !body) {
    DBGS("Failed to parse let_expression:\n");
    if (!binding) DBGS("  Missing binding\n");
    if (!body) DBGS("  Missing body\n");
    return nullptr;
  }
  
  return std::make_unique<LetExpressionAST>(
    getLocation(node, adaptor),
    std::move(binding),
    std::move(body)
  );
}

// AST Dump Implementation
void dumpASTNode(llvm::raw_ostream &os, const ASTNode *node, int indent = 0);

// Helper to print indentation
void printIndent(llvm::raw_ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << ANSIColors::faint << "| " << ANSIColors::reset;
  }
}

llvm::raw_ostream& ocamlc2::operator<<(llvm::raw_ostream &os, const ASTNode &node) {
  os << ANSIColors::reset;
  dumpASTNode(os, &node);
  return os;
}

void dumpASTNode(llvm::raw_ostream &os, const ASTNode *node, int indent) {
  if (!node) {
    printIndent(os, indent);
    os << "null\n";
    return;
  }
  
  printIndent(os, indent);
  
  switch (node->getKind()) {
    case ASTNode::Node_Number: {
      auto *num = static_cast<const NumberExprAST*>(node);
      os << "Number: " << num->getValue() << "\n";
      break;
    }
    case ASTNode::Node_ValuePath: {
      auto *valuePath = static_cast<const ValuePathAST*>(node);
      os << "ValuePath: ";
      bool first = true;
      for (const auto &part : valuePath->getPath()) {
        if (!first) os << ".";
        os << part;
        first = false;
      }
      os << "\n";
      break;
    }
    case ASTNode::Node_ConstructorPath: {
      auto *ctorPath = static_cast<const ConstructorPathAST*>(node);
      os << "ConstructorPath: ";
      bool first = true;
      for (const auto &part : ctorPath->getPath()) {
        if (!first) os << ".";
        os << part;
        first = false;
      }
      os << "\n";
      break;
    }
    case ASTNode::Node_TypeConstructorPath: {
      auto *typePath = static_cast<const TypeConstructorPathAST*>(node);
      os << "TypeConstructorPath: ";
      bool first = true;
      for (const auto &part : typePath->getPath()) {
        if (!first) os << ".";
        os << part;
        first = false;
      }
      os << "\n";
      break;
    }
    case ASTNode::Node_Application: {
      auto *app = static_cast<const ApplicationExprAST*>(node);
      os << "ApplicationExpr:\n";
      printIndent(os, indent + 1);
      os << "Function:\n";
      dumpASTNode(os, app->getFunction(), indent + 2);
      printIndent(os, indent + 1);
      os << "Arguments:\n";
      for (const auto &arg : app->getArguments()) {
        dumpASTNode(os, arg.get(), indent + 2);
      }
      break;
    }
    case ASTNode::Node_InfixExpression: {
      auto *infix = static_cast<const InfixExpressionAST*>(node);
      os << "InfixExpr: " << infix->getOperator() << "\n";
      printIndent(os, indent + 1);
      os << "LHS:\n";
      dumpASTNode(os, infix->getLHS(), indent + 2);
      printIndent(os, indent + 1);
      os << "RHS:\n";
      dumpASTNode(os, infix->getRHS(), indent + 2);
      break;
    }
    case ASTNode::Node_ParenthesizedExpression: {
      auto *paren = static_cast<const ParenthesizedExpressionAST*>(node);
      os << "ParenthesizedExpr:\n";
      dumpASTNode(os, paren->getExpression(), indent + 1);
      break;
    }
    case ASTNode::Node_MatchExpression: {
      auto *match = static_cast<const MatchExpressionAST*>(node);
      os << "MatchExpr:\n";
      printIndent(os, indent + 1);
      os << "Value:\n";
      dumpASTNode(os, match->getValue(), indent + 2);
      printIndent(os, indent + 1);
      os << "Cases:\n";
      for (const auto &matchCase : match->getCases()) {
        dumpASTNode(os, matchCase.get(), indent + 2);
      }
      break;
    }
    case ASTNode::Node_ForExpression: {
      auto *forExpr = static_cast<const ForExpressionAST*>(node);
      os << "ForExpr: " << forExpr->getLoopVar() << " = ";
      os << (forExpr->getIsDownto() ? "downto" : "to") << "\n";
      printIndent(os, indent + 1);
      os << "Start:\n";
      dumpASTNode(os, forExpr->getStartExpr(), indent + 2);
      printIndent(os, indent + 1);
      os << "End:\n";
      dumpASTNode(os, forExpr->getEndExpr(), indent + 2);
      printIndent(os, indent + 1);
      os << "Body:\n";
      dumpASTNode(os, forExpr->getBody(), indent + 2);
      break;
    }
    case ASTNode::Node_MatchCase: {
      auto *matchCase = static_cast<const MatchCaseAST*>(node);
      os << "MatchCase:\n";
      printIndent(os, indent + 1);
      os << "Pattern:\n";
      dumpASTNode(os, matchCase->getPattern(), indent + 2);
      printIndent(os, indent + 1);
      os << "Expression:\n";
      dumpASTNode(os, matchCase->getExpression(), indent + 2);
      break;
    }
    case ASTNode::Node_ValuePattern: {
      auto *valPattern = static_cast<const ValuePatternAST*>(node);
      os << "ValuePattern: " << valPattern->getName() << "\n";
      break;
    }
    case ASTNode::Node_ConstructorPattern: {
      auto *ctorPattern = static_cast<const ConstructorPatternAST*>(node);
      os << "ConstructorPattern:\n";
      printIndent(os, indent + 1);
      os << "Constructor:\n";
      dumpASTNode(os, ctorPattern->getConstructor(), indent + 2);
      if (!ctorPattern->getArguments().empty()) {
        printIndent(os, indent + 1);
        os << "Arguments:\n";
        for (const auto &arg : ctorPattern->getArguments()) {
          dumpASTNode(os, arg.get(), indent + 2);
        }
      }
      break;
    }
    case ASTNode::Node_TypedPattern: {
      auto *typedPattern = static_cast<const TypedPatternAST*>(node);
      os << "TypedPattern:\n";
      printIndent(os, indent + 1);
      os << "Pattern:\n";
      dumpASTNode(os, typedPattern->getPattern(), indent + 2);
      printIndent(os, indent + 1);
      os << "Type:\n";
      dumpASTNode(os, typedPattern->getType(), indent + 2);
      break;
    }
    case ASTNode::Node_TypeDefinition: {
      auto *typeDef = static_cast<const TypeDefinitionAST*>(node);
      os << "TypeDefinition:\n";
      for (const auto &binding : typeDef->getBindings()) {
        dumpASTNode(os, binding.get(), indent + 1);
      }
      break;
    }
    case ASTNode::Node_ValueDefinition: {
      auto *valueDef = static_cast<const ValueDefinitionAST*>(node);
      os << "ValueDefinition:\n";
      for (const auto &binding : valueDef->getBindings()) {
        dumpASTNode(os, binding.get(), indent + 1);
      }
      break;
    }
    case ASTNode::Node_ExpressionItem: {
      auto *exprItem = static_cast<const ExpressionItemAST*>(node);
      os << "ExpressionItem:\n";
      dumpASTNode(os, exprItem->getExpression(), indent + 1);
      break;
    }
    case ASTNode::Node_TypeBinding: {
      auto *typeBinding = static_cast<const TypeBindingAST*>(node);
      os << "TypeBinding: " << typeBinding->getName() << "\n";
      printIndent(os, indent + 1);
      os << "Definition:\n";
      dumpASTNode(os, typeBinding->getDefinition(), indent + 2);
      break;
    }
    case ASTNode::Node_VariantDeclaration: {
      auto *variantDecl = static_cast<const VariantDeclarationAST*>(node);
      os << "VariantDeclaration:\n";
      for (const auto &ctor : variantDecl->getConstructors()) {
        dumpASTNode(os, ctor.get(), indent + 1);
      }
      break;
    }
    case ASTNode::Node_ConstructorDeclaration: {
      auto *ctorDecl = static_cast<const ConstructorDeclarationAST*>(node);
      os << "ConstructorDeclaration: " << ctorDecl->getName();
      if (!ctorDecl->getOfTypes().empty()) {
        os << " of\n";
        for (const auto &type : ctorDecl->getOfTypes()) {
          dumpASTNode(os, type.get(), indent + 1);
        }
      } else {
        os << "\n";
      }
      break;
    }
    case ASTNode::Node_LetBinding: {
      auto *letBinding = static_cast<const LetBindingAST*>(node);
      os << "LetBinding: " << letBinding->getName() << "\n";
      
      if (!letBinding->getParameters().empty()) {
        printIndent(os, indent + 1);
        os << "Parameters:\n";
        for (const auto &param : letBinding->getParameters()) {
          dumpASTNode(os, param.get(), indent + 2);
        }
      }
      
      if (letBinding->getReturnType()) {
        printIndent(os, indent + 1);
        os << "ReturnType:\n";
        dumpASTNode(os, letBinding->getReturnType(), indent + 2);
      }
      
      printIndent(os, indent + 1);
      os << "Body:\n";
      dumpASTNode(os, letBinding->getBody(), indent + 2);
      break;
    }
    case ASTNode::Node_CompilationUnit: {
      auto *unit = static_cast<const CompilationUnitAST*>(node);
      os << "CompilationUnit:\n";
      for (const auto &item : unit->getItems()) {
        dumpASTNode(os, item.get(), indent + 1);
      }
      break;
    }
    case ASTNode::Node_LetExpression: {
      auto *letExpr = static_cast<const LetExpressionAST*>(node);
      os << "LetExpr:\n";
      printIndent(os, indent + 1);
      os << "Binding:\n";
      dumpASTNode(os, letExpr->getBinding(), indent + 2);
      printIndent(os, indent + 1);
      os << "Body:\n";
      dumpASTNode(os, letExpr->getBody(), indent + 2);
      break;
    }
  }
}
