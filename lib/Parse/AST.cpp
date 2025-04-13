#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include "ocamlc2/Support/LLVMCommon.h"
#include "ocamlc2/Support/Colors.h"
#include "ocamlc2/Parse/TSAdaptor.h"
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallSet.h>
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
    case Node_String: return "String";
    case Node_Boolean: return "Boolean";
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
    case Node_IfExpression: return "IfExpression";
    case Node_GuardedPattern: return "GuardedPattern";
    case Node_ListExpression: return "ListExpression";
    case Node_FunExpression: return "FunExpression";
    case Node_UnitExpression: return "UnitExpression";
    case Node_SignExpression: return "SignExpression";
    case Node_ArrayGetExpression: return "ArrayGetExpression";
    case Node_ArrayExpression: return "ArrayExpression";
    case Node_SequenceExpression: return "SequenceExpression";
  }
  return "";
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

  std::cerr << indentation << ANSIColors::cyan() << nodeType << ANSIColors::reset()
            << ": " << ANSIColors::italic() << text << ANSIColors::reset()
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
std::unique_ptr<StringExprAST> convertString(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<BooleanExprAST> convertBoolean(TSNode node, const TSTreeAdaptor &adaptor);
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
std::unique_ptr<IfExpressionAST> convertIfExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ListExpressionAST> convertListExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<FunExpressionAST> convertFunExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<UnitExpressionAST> convertUnitExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ValuePatternAST> convertValuePattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ConstructorPatternAST> convertConstructorPattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypedPatternAST> convertTypedPattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypeDefinitionAST> convertTypeDefinition(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ValueDefinitionAST> convertValueDefinition(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ExpressionItemAST> convertExpressionItem(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<TypeBindingAST> convertTypeBinding(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<VariantDeclarationAST> convertVariantDeclaration(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ConstructorDeclarationAST> convertConstructorDeclaration(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<LetBindingAST> convertLetBinding(TSNode node, const TSTreeAdaptor &adaptor, bool parentIsRec = false);
std::unique_ptr<CompilationUnitAST> convertCompilationUnit(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<GuardedPatternAST> convertGuardedPattern(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<SignExpressionAST> convertSignExpression(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ArrayGetExpressionAST> convertArrayGetExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<ArrayExpressionAST> convertArrayExpr(TSNode node, const TSTreeAdaptor &adaptor);
std::unique_ptr<SequenceExpressionAST> convertSequenceExpr(TSNode node, const TSTreeAdaptor &adaptor);

// Set of known/supported node types for better error reporting
llvm::DenseSet<llvm::StringRef> knownNodeTypes = {
    "compilation_unit", "number", "string", "boolean", "true", "false", "value_path", "constructor_path",
    "type_constructor_path", "application_expression", "infix_expression", "sign_expression", "sign_operator",
    "parenthesized_expression", "match_expression", "match_case",
    "value_pattern", "constructor_pattern", "typed_pattern", "type_definition",
    "value_definition", "expression_item", "type_binding", "variant_declaration",
    "constructor_declaration", "let_binding", "value_name", "type_constructor",
    "constructor_name", "add_operator", "subtract_operator", "multiply_operator",
    "division_operator", "concat_operator", "and_operator", "or_operator",
    "equal_operator", "parameter", "field_get_expression", "for_expression", 
    "let_expression", "if_expression", "if", "then", "else", "then_clause", "else_clause",
    "when", "guard", "in", "list_expression", "[", "]", "list",
    "fun_expression", "fun", "function_expression", "function", "unit",
    ";", ";;", "(", ")", ":", "=", "->", "|", "of", "with", "match", "type", "let",
    "do", "done", "to", "downto", "rec", "array_get_expression", "array_expression", "comment",
    "sequence_expression"
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
llvm::DenseMap<llvm::StringRef, llvm::StringRef> operatorMap = {
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
  else if (type == "string")
    return convertString(node, adaptor);
  else if (type == "boolean" || type == "true" || type == "false")
    return convertBoolean(node, adaptor);
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
  else if (type == "sign_expression" || type == "sign_operator")
    return convertSignExpression(node, adaptor);
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
  else if (type == "if_expression" || type == "if")
    return convertIfExpr(node, adaptor);
  else if (type == "list_expression")
    return convertListExpr(node, adaptor);
  else if (type == "array_expression")
    return convertArrayExpr(node, adaptor);
  else if (type == "fun_expression" || type == "fun" || 
           type == "function_expression" || type == "function")
    return convertFunExpr(node, adaptor);
  else if (type == "unit_expression" || type == "unit")
    return convertUnitExpr(node, adaptor);
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
    return convertLetBinding(node, adaptor, false);
  else if (type == "guard" || type == "when")
    return convertGuardedPattern(node, adaptor);
  else if (type == "sequence_expression")
    return convertSequenceExpr(node, adaptor);
  else if (type == "rec") {
    // For recursive bindings, we need to look at the parent node
    TSNode parent = ts_node_parent(node);
    if (!ts_node_is_null(parent)) {
      return convertNode(parent, adaptor);
    }
  }
  else if (type == "array_get_expression")
    return convertArrayGetExpr(node, adaptor);
  else if (type == "comment") {
    auto str = getNodeText(node, adaptor);
    if (str.contains("AJM")) {
      auto loc = getLocation(node, adaptor);
      llvm::errs() << loc.filename << ":" << loc.line << ":" << loc.column
                   << " " << str << "\n";
    }
    return nullptr;
  }
  else {
    if (knownNodeTypes.find(type.str()) == knownNodeTypes.end()) {
      // Unknown node type, log it for debugging
      DBGS("Unknown node type: " << type << "\n");
    }
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

std::unique_ptr<StringExprAST> convertString(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "string") {
    return nullptr;
  }
  
  std::string value = getNodeText(node, adaptor);
  // Remove the quotes from the string literal
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    value = value.substr(1, value.size() - 2);
  }
  
  return std::make_unique<StringExprAST>(getLocation(node, adaptor), value);
}

std::unique_ptr<BooleanExprAST> convertBoolean(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  bool isBool = (std::string(nodeType) == "boolean");
  bool isTrue = (std::string(nodeType) == "true");
  bool isFalse = (std::string(nodeType) == "false");
  
  if (!isBool && !isTrue && !isFalse) {
    return nullptr;
  }
  
  // If we have the boolean wrapper node, look at its children for the actual value
  if (isBool) {
    auto children = childrenNodes(node);
    for (auto [childType, child] : children) {
      if (childType == "true" || childType == "false") {
        return convertBoolean(child, adaptor);
      }
    }
    return nullptr;
  }
  
  // We're directly at a true/false node
  bool value = isTrue; // true or false
  return std::make_unique<BooleanExprAST>(getLocation(node, adaptor), value);
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
  bool hasGuard = false;
  std::unique_ptr<ASTNode> guardExpr = nullptr;
  
  for (auto [type, child] : children) {
    if (type == "->") {
      foundArrow = true;
      continue;
    } else if (type == "guard") {
      hasGuard = true;
      // Handle guard separately
      auto guardChildren = childrenNodes(child);
      for (auto [guardType, guardChild] : guardChildren) {
        if (guardType != "when") {
          guardExpr = convertNode(guardChild, adaptor);
        }
      }
      continue;
    }
    
    if (!foundArrow && !pattern) {
      pattern = convertNode(child, adaptor);
    } else if (foundArrow && !expr) {
      expr = convertNode(child, adaptor);
    }
  }
  
  // If we have a guard, create a GuardedPatternAST
  if (hasGuard && pattern && guardExpr) {
    pattern = std::make_unique<GuardedPatternAST>(
      getLocation(node, adaptor),
      std::move(pattern),
      std::move(guardExpr)
    );
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

std::unique_ptr<LetBindingAST> convertLetBinding(TSNode node, const TSTreeAdaptor &adaptor, bool parentIsRec) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "let_binding") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::string name;
  std::vector<std::unique_ptr<ASTNode>> parameters;
  std::unique_ptr<TypeConstructorPathAST> returnType = nullptr;
  std::unique_ptr<ASTNode> body = nullptr;
  bool isRec = parentIsRec;  // Start with the parent's isRec value
  bool hasUnitPattern = false;
  
  // Also check for "rec" keyword within this node to identify recursive bindings
  for (auto [type, child] : children) {
    if (type == "rec") {
      isRec = true;
      DBGS("Found recursive keyword 'rec', setting isRec = true\n");
      break;
    }
  }
  
  // First pass: look for name, parameters, unit pattern, and simple expression
  for (auto [type, child] : children) {
    if (type == "value_name") {
      name = getNodeText(child, adaptor);
    } else if (type == "parameter") {
      auto param = convertNode(child, adaptor);
      if (param) {
        parameters.push_back(std::move(param));
      } else {
        // Try to extract value patterns directly from parameter node
        auto paramChildren = childrenNodes(child);
        for (auto [paramType, paramChild] : paramChildren) {
          if (paramType == "value_pattern") {
            auto valPattern = convertValuePattern(paramChild, adaptor);
            if (valPattern) {
              parameters.push_back(std::move(valPattern));
            }
          }
        }
      }
    } else if (type == "unit") {
      // This is a unit pattern like "let () = ..."
      hasUnitPattern = true;
      DBGS("Found unit pattern in let binding\n");
    } else if (type == "(") {
      // Check if this might be part of a unit pattern
      // Look for a closing parenthesis with nothing in between
      size_t idx = 0;
      for (size_t i = 0; i < children.size(); i++) {
        if (children[i].first == type) {
          idx = i;
          break;
        }
      }
      
      if (idx + 1 < children.size() && children[idx + 1].first == ")") {
        hasUnitPattern = true;
        DBGS("Found unit pattern (() as LHS in let binding\n");
      }
    } else if (type == "type_constructor_path") {
      returnType = convertTypeConstructorPath(child, adaptor);
    } else if (type == "=" && !body) {
      // The next child after '=' is likely the body
      size_t idx = 0;
      for (size_t i = 0; i < children.size(); i++) {
        if (children[i].first == type) {
          idx = i;
          break;
        }
      }
      
      // Look at the next node after '='
      if (idx + 1 < children.size()) {
        body = convertNode(children[idx + 1].second, adaptor);
      }
    }
  }
  
  // If we don't have a body yet, try a different approach
  if (!body) {
    for (auto [type, child] : children) {
      if (type != "value_name" && type != "parameter" && 
          type != ":" && type != "=" && type != "type_constructor_path" && 
          type != "unit" && type != "(" && type != ")") {
        auto possibleBody = convertNode(child, adaptor);
        if (possibleBody) {
          body = std::move(possibleBody);
          break;
        }
      }
    }
  }
  
  // For recursive bindings, try to handle special cases
  if (name.empty() && !hasUnitPattern) {
    for (auto [type, child] : children) {
      if (type == "rec") {
        isRec = true;
        break;
      }
    }
    
    if (isRec) {
      // For recursive bindings, look for value name after "rec"
      size_t recIdx = 0;
      for (size_t i = 0; i < children.size(); i++) {
        if (children[i].first == "rec") {
          recIdx = i;
          break;
        }
      }
      
      // Try to find the name after "rec"
      for (size_t i = recIdx + 1; i < children.size(); i++) {
        if (children[i].first == "value_name") {
          name = getNodeText(children[i].second, adaptor);
          break;
        }
      }
    }
  }
  
  // Debug what we found
  DBGS("Parsing let_binding: \n");
  DBGS("  Has unit pattern: " << (hasUnitPattern ? "yes" : "no") << "\n");
  DBGS("  Name: " << (name.empty() ? "missing" : name) << "\n");
  DBGS("  Parameters: " << parameters.size() << "\n");
  DBGS("  Return type: " << (returnType ? "found" : "missing") << "\n");
  DBGS("  Body: " << (body ? "found" : "missing") << "\n");
  DBGS("  Is recursive: " << (isRec ? "yes" : "no") << "\n");
  
  // Special handling for unit binding patterns (let () = ...)
  if (hasUnitPattern) {
    if (body) {
      // This is a unit binding like "let () = ..."
      // Use a special name for unit bindings
      return std::make_unique<LetBindingAST>(
        getLocation(node, adaptor),
        "unit!", // Special name for unit bindings
        std::move(parameters),
        std::move(returnType),
        std::move(body),
        isRec
      );
    } else {
      DBGS("Failed to parse unit let_binding: missing body\n");
      return nullptr;
    }
  }
  
  if (name.empty() && !hasUnitPattern) {
    DBGS("Failed to parse let_binding:\n");
    DBGS("  Missing name and not a unit pattern\n");
    
    // As a last resort for unit value bindings (let () = ...)
    bool hasOpenParen = false;
    bool hasCloseParen = false;
    for (auto [type, child] : children) {
      if (type == "(") {
        hasOpenParen = true;
      } else if (type == ")") {
        hasCloseParen = true;
      }
    }
    
    if (hasOpenParen && hasCloseParen && children.size() <= 5) {
      // This is likely a unit binding pattern that wasn't detected earlier
      DBGS("Detected likely unit pattern from parentheses\n");
      return std::make_unique<LetBindingAST>(
        getLocation(node, adaptor),
        "unit!", // Special name for unit bindings
        std::move(parameters),
        std::move(returnType),
        body ? std::move(body) : std::make_unique<UnitExpressionAST>(getLocation(node, adaptor)),
        isRec
      );
    }
    
    return nullptr;
  }
  
  if (!body) {
    DBGS("Failed to parse let_binding:\n");
    DBGS("  Missing body\n");
    return nullptr;
  }
  
  return std::make_unique<LetBindingAST>(
    getLocation(node, adaptor),
    name,
    std::move(parameters),
    std::move(returnType),
    std::move(body),
    isRec
  );
}

std::unique_ptr<ValueDefinitionAST> convertValueDefinition(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "value_definition") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<LetBindingAST>> bindings;
  bool isRec = false;
  
  // Check for 'rec' keyword at the value_definition level
  for (auto [type, child] : children) {
    if (type == "rec") {
      isRec = true;
      DBGS("Found recursive keyword at value_definition level, setting isRec = true\n");
      break;
    }
  }
  
  for (auto [type, child] : children) {
    if (type == "let_binding") {
      auto binding = convertLetBinding(child, adaptor, isRec);
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
        type == "expression_item" ||
        type == "comment") {
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
  
  // Debug children for diagnosis
  DBGS("Let expression children:\n");
  for (auto [type, child] : children) {
    DBGS("  Type: " << type.str() << "\n");
  }
  
  // First pass: look for binding and 'in' keyword
  for (auto [type, child] : children) {
    if (type == "let_binding") {
      // Direct let_binding
      binding = convertLetBinding(child, adaptor, false);
    } else if (type == "value_definition") {
      // Value definition containing let_binding
      auto valueDef = convertValueDefinition(child, adaptor);
      if (valueDef) {
        binding = std::move(valueDef);
      } else {
        // Try to extract let_binding directly if value_definition conversion failed
        auto defChildren = childrenNodes(child);
        for (auto [defType, defChild] : defChildren) {
          if (defType == "let_binding") {
            binding = convertLetBinding(defChild, adaptor, false);
            if (binding) break;
          }
        }
      }
    } else if (type == "in") {
      foundIn = true;
    } else if (foundIn && !body) {
      // This should be the body (anything after 'in')
      body = convertNode(child, adaptor);
    }
  }
  
  // If not found yet, try a second approach - get the next node after 'in'
  if (foundIn && !body) {
    size_t inIdx = 0;
    for (size_t i = 0; i < children.size(); i++) {
      if (children[i].first == "in") {
        inIdx = i;
        break;
      }
    }
    
    // Look for body after 'in'
    if (inIdx + 1 < children.size()) {
      body = convertNode(children[inIdx + 1].second, adaptor);
    }
  }
  
  // If we didn't find a binding yet, try a more general approach
  if (!binding) {
    for (auto [type, child] : children) {
      if (type == "let") {
        // Check if next node could be a binding
        size_t letIdx = 0;
        for (size_t i = 0; i < children.size(); i++) {
          if (children[i].first == "let") {
            letIdx = i;
            break;
          }
        }
        
        // Try to find let binding after 'let'
        if (letIdx + 1 < children.size()) {
          auto nextChild = children[letIdx + 1].second;
          if (std::string(ts_node_type(nextChild)) == "let_binding") {
            binding = convertLetBinding(nextChild, adaptor, false);
          }
        }
      } else if (type != "in" && !foundIn && !binding) {
        auto possibleBinding = convertNode(child, adaptor);
        if (possibleBinding) {
          binding = std::move(possibleBinding);
        }
      }
    }
  }
  
  // Debug what we found
  DBGS("Parsing let_expression: \n");
  DBGS("  Binding: " << (binding ? "found" : "missing") << "\n");
  DBGS("  Body: " << (body ? "found" : "missing") << "\n");
  
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

std::unique_ptr<IfExpressionAST> convertIfExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  bool isIfExpr = (std::string(nodeType) == "if_expression");
  bool isIfKeyword = (std::string(nodeType) == "if");
  
  if (!isIfExpr && !isIfKeyword) {
    return nullptr;
  }
  
  // If we have the 'if' keyword, we need to look at its parent
  TSNode targetNode = isIfKeyword ? ts_node_parent(node) : node;
  if (ts_node_is_null(targetNode) || 
      (isIfKeyword && std::string(ts_node_type(targetNode)) != "if_expression")) {
    return nullptr;
  }
  
  auto children = childrenNodes(targetNode);
  std::unique_ptr<ASTNode> condition = nullptr;
  std::unique_ptr<ASTNode> thenBranch = nullptr;
  std::unique_ptr<ASTNode> elseBranch = nullptr;
  
  // First pass to identify clause nodes directly
  for (auto [type, child] : children) {
    if (type == "if") {
      continue; // Skip 'if' keyword
    } else if (type == "boolean" || type == "true" || type == "false") {
      // This is likely the condition
      condition = convertBoolean(child, adaptor);
    } else if (type == "then_clause") {
      // Extract expression from then_clause
      auto thenChildren = childrenNodes(child);
      for (auto [thenType, thenChild] : thenChildren) {
        if (thenType != "then") {
          thenBranch = convertNode(thenChild, adaptor);
          if (thenBranch) break;
        }
      }
    } else if (type == "else_clause") {
      // Extract expression from else_clause
      auto elseChildren = childrenNodes(child);
      for (auto [elseType, elseChild] : elseChildren) {
        if (elseType != "else") {
          // Check if there's a nested if expression in the else clause
          if (elseType == "if_expression") {
            // Handle nested if expression
            elseBranch = convertIfExpr(elseChild, adaptor);
          } else {
            elseBranch = convertNode(elseChild, adaptor);
          }
          if (elseBranch) break;
        }
      }
    } else if (!condition) {
      // First non-keyword node is likely the condition
      condition = convertNode(child, adaptor);
    }
  }
  
  // If we still don't have the required parts, try a deeper search
  if (!condition || !thenBranch) {
    // Second-chance pass looking for the components
    for (auto [type, child] : children) {
      if (!condition && type != "if" && type != "then" && type != "else" && 
          type != "then_clause" && type != "else_clause") {
        condition = convertNode(child, adaptor);
      } else if (!thenBranch && (type == "then" || type == "then_clause")) {
        // Look for the expression after 'then'
        size_t idx = 0;
        for (size_t i = 0; i < children.size(); i++) {
          if (children[i].first == type) {
            idx = i;
            break;
          }
        }
        
        // The next node after 'then' should be the then branch
        if (idx + 1 < children.size()) {
          thenBranch = convertNode(children[idx + 1].second, adaptor);
        }
      } else if (!elseBranch && (type == "else" || type == "else_clause")) {
        // Look for the expression after 'else'
        size_t idx = 0;
        for (size_t i = 0; i < children.size(); i++) {
          if (children[i].first == type) {
            idx = i;
            break;
          }
        }
        
        // The next node after 'else' should be the else branch
        if (idx + 1 < children.size()) {
          // Check if the next node is an if expression
          auto nextType = children[idx + 1].first;
          auto nextNode = children[idx + 1].second;
          if (nextType == "if_expression" || nextType == "if") {
            elseBranch = convertIfExpr(nextNode, adaptor);
          } else {
            elseBranch = convertNode(nextNode, adaptor);
          }
        }
      }
    }
  }
  
  // Debug what we found
  DBGS("Parsing if_expression: \n");
  DBGS("  Condition: " << (condition ? "found" : "missing") << "\n");
  DBGS("  Then branch: " << (thenBranch ? "found" : "missing") << "\n");
  DBGS("  Else branch: " << (elseBranch ? "found" : "missing") << "\n");
  
  if (!condition || !thenBranch) {
    DBGS("Failed to parse if_expression:\n");
    if (!condition) DBGS("  Missing condition\n");
    if (!thenBranch) DBGS("  Missing then branch\n");
    return nullptr;
  }
  
  return std::make_unique<IfExpressionAST>(
    getLocation(targetNode, adaptor),
    std::move(condition),
    std::move(thenBranch),
    std::move(elseBranch)
  );
}

std::unique_ptr<GuardedPatternAST> convertGuardedPattern(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  bool isGuard = (std::string(nodeType) == "guard");
  bool isWhen = (std::string(nodeType) == "when");
  
  if (!isGuard && !isWhen) {
    return nullptr;
  }
  
  // For a 'when' node, look at the parent which should be a 'guard'
  if (isWhen) {
    TSNode parent = ts_node_parent(node);
    if (ts_node_is_null(parent) || std::string(ts_node_type(parent)) != "guard") {
      return nullptr;
    }
    node = parent;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> pattern = nullptr;
  std::unique_ptr<ASTNode> guard = nullptr;
  
  // Find the pattern and guard expression
  bool foundWhen = false;
  
  // First, try to extract pattern before the 'when' and the guard expression after it
  for (auto [type, child] : children) {
    if (type == "when") {
      foundWhen = true;
      continue;
    }
    
    if (!foundWhen) {
      // This should be the pattern
      if (!pattern) {
        pattern = convertNode(child, adaptor);
      }
    } else {
      // This should be the guard expression
      if (!guard) {
        guard = convertNode(child, adaptor);
      }
    }
  }
  
  // If not successful, try a different approach - look at the parent match_case node
  if (!pattern || !guard) {
    TSNode matchCase = ts_node_parent(node);
    if (!ts_node_is_null(matchCase) && std::string(ts_node_type(matchCase)) == "match_case") {
      auto matchCaseChildren = childrenNodes(matchCase);
      
      for (auto [type, child] : matchCaseChildren) {
        if (type != "guard" && type != "->" && !pattern) {
          pattern = convertNode(child, adaptor);
        } else if (type == "guard") {
          auto guardChildren = childrenNodes(child);
          for (auto [guardType, guardChild] : guardChildren) {
            if (guardType != "when") {
              guard = convertNode(guardChild, adaptor);
              break;
            }
          }
        }
      }
    }
  }
  
  if (!pattern || !guard) {
    DBGS("Failed to parse guarded_pattern:\n");
    if (!pattern) DBGS("  Missing pattern\n");
    if (!guard) DBGS("  Missing guard expression\n");
    return nullptr;
  }
  
  return std::make_unique<GuardedPatternAST>(
    getLocation(node, adaptor),
    std::move(pattern),
    std::move(guard)
  );
}

std::unique_ptr<SignExpressionAST> convertSignExpression(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  bool isSignExpr = (std::string(nodeType) == "sign_expression");
  bool isSignOperator = (std::string(nodeType) == "sign_operator");
  
  if (!isSignExpr && !isSignOperator) {
    return nullptr;
  }
  
  // If we have a sign_operator node, look at its parent which should be a sign_expression
  TSNode targetNode = isSignOperator ? ts_node_parent(node) : node;
  if (ts_node_is_null(targetNode) || 
      (isSignOperator && std::string(ts_node_type(targetNode)) != "sign_expression")) {
    return nullptr;
  }
  
  auto children = childrenNodes(targetNode);
  std::string op;
  std::unique_ptr<ASTNode> operand = nullptr;
  
  // First pass to identify the operator and operand
  for (auto [type, child] : children) {
    if (type == "sign_operator") {
      op = getNodeText(child, adaptor);
    } else if (!operand) {
      // The first non-operator node should be the operand
      operand = convertNode(child, adaptor);
    }
  }
  
  // If we didn't find the operand yet, try looking for specific node types
  if (!operand) {
    for (auto [type, child] : children) {
      if (type != "sign_operator" && type != "sign_expression") {
        operand = convertNode(child, adaptor);
        if (operand) break;
      }
    }
  }
  
  if (op.empty() || !operand) {
    DBGS("Failed to parse sign_expression:\n");
    if (op.empty()) DBGS("  Missing operator\n");
    if (!operand) DBGS("  Missing operand\n");
    return nullptr;
  }
  
  return std::make_unique<SignExpressionAST>(
    getLocation(targetNode, adaptor),
    op,
    std::move(operand)
  );
}

std::unique_ptr<ListExpressionAST> convertListExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  bool isListExpr = (std::string(nodeType) == "list_expression");
  bool isListStart = (std::string(nodeType) == "[");
  
  if (!isListExpr && !isListStart) {
    return nullptr;
  }
  
  // If we have the '[' token, look at its parent which should be a list_expression
  TSNode targetNode = isListStart ? ts_node_parent(node) : node;
  if (ts_node_is_null(targetNode) || std::string(ts_node_type(targetNode)) != "list_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(targetNode);
  std::vector<std::unique_ptr<ASTNode>> elements;
  
  // Skip list delimiters '[' and ']' and semicolons ';'
  for (auto [type, child] : children) {
    if (type != "[" && type != "]" && type != ";") {
      auto element = convertNode(child, adaptor);
      if (element) {
        elements.push_back(std::move(element));
      }
    }
  }
  
  return std::make_unique<ListExpressionAST>(
    getLocation(targetNode, adaptor),
    std::move(elements)
  );
}

std::unique_ptr<FunExpressionAST> convertFunExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  bool isFunExpr = (std::string(nodeType) == "fun_expression");
  bool isFunKeyword = (std::string(nodeType) == "fun");
  bool isFunctionExpr = (std::string(nodeType) == "function_expression");
  bool isFunctionKeyword = (std::string(nodeType) == "function");
  
  if (!isFunExpr && !isFunKeyword && !isFunctionExpr && !isFunctionKeyword) {
    return nullptr;
  }
  
  // If we have just the keyword, look at its parent
  TSNode targetNode = (isFunKeyword || isFunctionKeyword) ? ts_node_parent(node) : node;
  if (ts_node_is_null(targetNode)) {
    return nullptr;
  }
  
  auto children = childrenNodes(targetNode);
  std::vector<std::unique_ptr<ASTNode>> parameters;
  std::unique_ptr<ASTNode> body = nullptr;
  
  bool foundArrow = false;
  
  // Handle 'fun' expressions (e.g., fun x -> x + 1)
  if (isFunExpr || isFunKeyword) {
    for (auto [type, child] : children) {
      if (type == "fun") {
        continue; // Skip the 'fun' keyword
      } else if (type == "->") {
        foundArrow = true;
        continue;
      } else if (type == "parameter") {
        auto param = convertNode(child, adaptor);
        if (param) {
          parameters.push_back(std::move(param));
        }
      } else if (foundArrow && !body) {
        body = convertNode(child, adaptor);
      }
    }
  }
  // Handle 'function' expressions which use pattern matching
  else if (isFunctionExpr || isFunctionKeyword) {
    // For function expressions, we treat the first match case as the body
    // since 'function' is basically a shorthand for 'fun x -> match x with ...'
    for (auto [type, child] : children) {
      if (type == "function") {
        continue; // Skip 'function' keyword
      } else if (type == "|") {
        continue; // Skip '|' token
      } else if (type == "match_case") {
        // Use the first match case as our body
        if (!body) {
          body = convertMatchCase(child, adaptor);
        }
      }
    }
    
    // For 'function', we add an implicit parameter (which is matched in the body)
    parameters.push_back(std::make_unique<ValuePatternAST>(
      getLocation(targetNode, adaptor),
      "_implicit_fun_param" // Use a special name for the implicit parameter
    ));
  }
  
  if (parameters.empty() || !body) {
    DBGS("Failed to parse fun_expression:\n");
    if (parameters.empty()) DBGS("  Missing parameters\n");
    if (!body) DBGS("  Missing body\n");
    return nullptr;
  }
  
  return std::make_unique<FunExpressionAST>(
    getLocation(targetNode, adaptor),
    std::move(parameters),
    std::move(body)
  );
}

std::unique_ptr<UnitExpressionAST> convertUnitExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "unit") {
    return nullptr;
  }
  
  // Verify that the node is actually a unit () by checking its children
  auto children = childrenNodes(node);
  bool hasOpenParen = false;
  bool hasCloseParen = false;
  
  for (auto [type, child] : children) {
    if (type == "(") {
      hasOpenParen = true;
    } else if (type == ")") {
      hasCloseParen = true;
    }
  }
  
  if (hasOpenParen && hasCloseParen) {
    return std::make_unique<UnitExpressionAST>(getLocation(node, adaptor));
  }
  
  return nullptr;
}

std::unique_ptr<ArrayGetExpressionAST> convertArrayGetExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "array_get_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::unique_ptr<ASTNode> array = nullptr;
  std::unique_ptr<ASTNode> index = nullptr;
  
  // First find the array (first child before the dot)
  for (auto [type, child] : children) {
    if (type != "." && type != "(" && type != ")" && !array) {
      array = convertNode(child, adaptor);
      break;
    }
  }
  
  // Then find the index (between parentheses)
  bool foundLeftParen = false;
  for (auto [type, child] : children) {
    if (type == "(") {
      foundLeftParen = true;
      continue;
    }
    
    if (foundLeftParen && type != ")") {
      index = convertNode(child, adaptor);
      break;
    }
  }
  
  if (!array || !index) {
    DBGS("Failed to parse array_get_expression:\n");
    if (!array) DBGS("  Missing array\n");
    if (!index) DBGS("  Missing index\n");
    return nullptr;
  }
  
  return std::make_unique<ArrayGetExpressionAST>(
    getLocation(node, adaptor),
    std::move(array),
    std::move(index)
  );
}

std::unique_ptr<ArrayExpressionAST> convertArrayExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "array_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<ASTNode>> elements;
  
  // Skip array delimiters [| and |] and semicolons ;
  for (auto [type, child] : children) {
    if (type != "[|" && type != "|]" && type != ";") {
      auto element = convertNode(child, adaptor);
      if (element) {
        elements.push_back(std::move(element));
      }
    }
  }
  
  return std::make_unique<ArrayExpressionAST>(
    getLocation(node, adaptor),
    std::move(elements)
  );
}

std::unique_ptr<SequenceExpressionAST> convertSequenceExpr(TSNode node, const TSTreeAdaptor &adaptor) {
  const char* nodeType = ts_node_type(node);
  if (std::string(nodeType) != "sequence_expression") {
    return nullptr;
  }
  
  auto children = childrenNodes(node);
  std::vector<std::unique_ptr<ASTNode>> expressions;
  
  // Process each expression in the sequence, skipping semicolons
  for (auto [type, child] : children) {
    if (type != ";") {
      auto expr = convertNode(child, adaptor);
      if (expr) {
        expressions.push_back(std::move(expr));
      }
    }
  }
  
  if (expressions.empty()) {
    DBGS("Failed to parse sequence_expression:\n");
    DBGS("  No expressions found\n");
    return nullptr;
  }
  
  return std::make_unique<SequenceExpressionAST>(
    getLocation(node, adaptor),
    std::move(expressions)
  );
}

// AST Dump Implementation
void dumpASTNode(llvm::raw_ostream &os, const ASTNode *node, int indent = 0);

// Helper to print indentation
void printIndent(llvm::raw_ostream &os, int indent) {
  for (int i = 0; i < indent; ++i) {
    os << ANSIColors::faint() << "| " << ANSIColors::reset();
  }
}

llvm::raw_ostream& ocamlc2::operator<<(llvm::raw_ostream &os, const ASTNode &node) {
  os << ANSIColors::reset();
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
    case ASTNode::Node_String: {
      auto *str = static_cast<const StringExprAST*>(node);
      os << "String: \"" << str->getValue() << "\"\n";
      break;
    }
    case ASTNode::Node_Boolean: {
      auto *boolean = static_cast<const BooleanExprAST*>(node);
      os << "Boolean: " << (boolean->getValue() ? "true" : "false") << "\n";
      break;
    }
    case ASTNode::Node_SignExpression: {
      auto *signExpr = static_cast<const SignExpressionAST*>(node);
      os << "SignExpr: " << signExpr->getOperator() << "\n";
      
      printIndent(os, indent + 1);
      os << "Operand:\n";
      dumpASTNode(os, signExpr->getOperand(), indent + 2);
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
      os << "LetBinding: " << (letBinding->getIsRecursive() ? "rec " : "")
         << letBinding->getName() << "\n";

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
    case ASTNode::Node_IfExpression: {
      auto *ifExpr = static_cast<const IfExpressionAST*>(node);
      os << "IfExpr:\n";
      printIndent(os, indent + 1);
      os << "Condition:\n";
      dumpASTNode(os, ifExpr->getCondition(), indent + 2);
      printIndent(os, indent + 1);
      os << "Then:\n";
      dumpASTNode(os, ifExpr->getThenBranch(), indent + 2);
      if (ifExpr->hasElseBranch()) {
        printIndent(os, indent + 1);
        os << "Else:\n";
        dumpASTNode(os, ifExpr->getElseBranch(), indent + 2);
      }
      break;
    }
    case ASTNode::Node_GuardedPattern: {
      auto *guarded = static_cast<const GuardedPatternAST*>(node);
      os << "GuardedPattern:\n";
      printIndent(os, indent + 1);
      os << "Pattern:\n";
      dumpASTNode(os, guarded->getPattern(), indent + 2);
      printIndent(os, indent + 1);
      os << "Guard:\n";
      dumpASTNode(os, guarded->getGuard(), indent + 2);
      break;
    }
    case ASTNode::Node_ListExpression: {
      auto *list = static_cast<const ListExpressionAST*>(node);
      os << "ListExpr: [";
      bool first = true;
      for (size_t i = 0; i < list->getNumElements(); ++i) {
        if (!first) os << "; ";
        first = false;
        const auto *element = list->getElement(i);
        if (auto *numExpr = llvm::dyn_cast<NumberExprAST>(element)) {
          os << numExpr->getValue();
        } else if (auto *strExpr = llvm::dyn_cast<StringExprAST>(element)) {
          os << "\"" << strExpr->getValue() << "\"";
        } else {
          os << "...";
        }
      }
      os << "]\n";
      if (list->getNumElements() > 0) {
        printIndent(os, indent + 1);
        os << "Elements:\n";
        for (size_t i = 0; i < list->getNumElements(); ++i) {
          dumpASTNode(os, list->getElement(i), indent + 2);
        }
      }
      break;
    }
    case ASTNode::Node_FunExpression: {
      auto *fun = static_cast<const FunExpressionAST*>(node);
      os << "FunExpr:\n";
      printIndent(os, indent + 1);
      os << "Parameters:\n";
      for (const auto &param : fun->getParameters()) {
        dumpASTNode(os, param.get(), indent + 2);
      }
      printIndent(os, indent + 1);
      os << "Body:\n";
      dumpASTNode(os, fun->getBody(), indent + 2);
      break;
    }
    case ASTNode::Node_UnitExpression:
      os << "UnitExpression:\n";
      break;
    case ASTNode::Node_ArrayGetExpression: {
      auto *arrayGetExpr = static_cast<const ArrayGetExpressionAST*>(node);
      os << "ArrayGetExpr:\n";
      
      printIndent(os, indent + 1);
      os << "Array:\n";
      dumpASTNode(os, arrayGetExpr->getArray(), indent + 2);
      
      printIndent(os, indent + 1);
      os << "Index:\n";
      dumpASTNode(os, arrayGetExpr->getIndex(), indent + 2);
      break;
    }
    case ASTNode::Node_ArrayExpression: {
      auto *array = static_cast<const ArrayExpressionAST*>(node);
      os << "ArrayExpr: [|";
      bool first = true;
      for (size_t i = 0; i < array->getNumElements(); ++i) {
        if (!first) os << "; ";
        first = false;
        const auto *element = array->getElement(i);
        if (auto *numExpr = llvm::dyn_cast<NumberExprAST>(element)) {
          os << numExpr->getValue();
        } else if (auto *strExpr = llvm::dyn_cast<StringExprAST>(element)) {
          os << "\"" << strExpr->getValue() << "\"";
        } else {
          os << "...";
        }
      }
      os << "|]\n";
      if (array->getNumElements() > 0) {
        printIndent(os, indent + 1);
        os << "Elements:\n";
        for (size_t i = 0; i < array->getNumElements(); ++i) {
          dumpASTNode(os, array->getElement(i), indent + 2);
        }
      }
      break;
    }
    case ASTNode::Node_SequenceExpression: {
      auto *seqExpr = static_cast<const SequenceExpressionAST*>(node);
      os << "SequenceExpr:\n";
      printIndent(os, indent + 1);
      os << "Expressions:\n";
      for (size_t i = 0; i < seqExpr->getNumExpressions(); ++i) {
        dumpASTNode(os, seqExpr->getExpression(i), indent + 2);
      }
      break;
    }
  }
}
