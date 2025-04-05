#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Location.h>
namespace ocamlc2 {

class ASTNode;
struct Location;

std::unique_ptr<ASTNode> parse(const std::string &source);
llvm::raw_ostream& operator<<(llvm::raw_ostream &os, const ASTNode &node);

/// Location in source code
struct Location {
  std::string filename;
  unsigned line, column;
};

/// Base class for all AST nodes
class ASTNode {
public:
  enum ASTNodeKind {
    // Expressions
    Node_Number,
    Node_ValuePath,
    Node_ConstructorPath,
    Node_TypeConstructorPath,
    Node_Application,
    Node_InfixExpression,
    Node_ParenthesizedExpression,
    Node_MatchExpression,
    
    // Patterns
    Node_ValuePattern,
    Node_ConstructorPattern,
    Node_TypedPattern,
    
    // Declarations
    Node_TypeDefinition,
    Node_ValueDefinition,
    Node_ExpressionItem,
    
    // Type related
    Node_TypeBinding,
    Node_VariantDeclaration,
    Node_ConstructorDeclaration,
    
    // Others
    Node_MatchCase,
    Node_LetBinding,
    Node_CompilationUnit,
  };

  ASTNode(ASTNodeKind kind, Location loc = {}) : kind(kind), location(std::move(loc)) {}
  virtual ~ASTNode() = default;
  
  ASTNodeKind getKind() const { return kind; }
  const Location& loc() const { return location; }
  mlir::Location getMLIRLocation(mlir::MLIRContext &context) const;
  static llvm::StringRef getName(ASTNodeKind kind);
  static llvm::StringRef getName(const ASTNode &node);

private:
  const ASTNodeKind kind;
  Location location;
};

using ASTNodeList = std::vector<std::unique_ptr<ASTNode>>;

/// Number literal expression (e.g., 1, 2.5)
class NumberExprAST : public ASTNode {
  std::string value; // Using string to preserve exact source representation
public:
  NumberExprAST(Location loc, std::string value)
    : ASTNode(Node_Number, std::move(loc)), value(std::move(value)) {}
  
  const std::string& getValue() const { return value; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Number;
  }
};

/// Value path expression (e.g., x, List.length)
class ValuePathAST : public ASTNode {
  std::vector<std::string> path;
public:
  ValuePathAST(Location loc, std::vector<std::string> path)
    : ASTNode(Node_ValuePath, std::move(loc)), path(std::move(path)) {}
  
  const std::vector<std::string>& getPath() const { return path; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ValuePath;
  }
};

/// Constructor path (e.g., A, Some)
class ConstructorPathAST : public ASTNode {
  std::vector<std::string> path;
public:
  ConstructorPathAST(Location loc, std::vector<std::string> path)
    : ASTNode(Node_ConstructorPath, std::move(loc)), path(std::move(path)) {}
  
  const std::vector<std::string>& getPath() const { return path; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ConstructorPath;
  }
};

/// Type constructor path (e.g., int, list)
class TypeConstructorPathAST : public ASTNode {
  std::vector<std::string> path;
public:
  TypeConstructorPathAST(Location loc, std::vector<std::string> path)
    : ASTNode(Node_TypeConstructorPath, std::move(loc)), path(std::move(path)) {}
  
  const std::vector<std::string>& getPath() const { return path; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_TypeConstructorPath;
  }
};

/// Function application (e.g., f x, g (x, y))
class ApplicationExprAST : public ASTNode {
  std::unique_ptr<ASTNode> function;
  std::vector<std::unique_ptr<ASTNode>> arguments;
public:
  ApplicationExprAST(Location loc, std::unique_ptr<ASTNode> function,
                     std::vector<std::unique_ptr<ASTNode>> arguments)
    : ASTNode(Node_Application, std::move(loc)),
      function(std::move(function)),
      arguments(std::move(arguments)) {}
  
  const ASTNode* getFunction() const { return function.get(); }
  const std::vector<std::unique_ptr<ASTNode>>& getArguments() const { return arguments; }
  size_t getNumArguments() const { return arguments.size(); }
  ASTNode *getArgument(size_t index) const { return arguments[index].get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Application;
  }
};

/// Infix expression (e.g., a + b, x :: xs)
class InfixExpressionAST : public ASTNode {
  std::unique_ptr<ASTNode> lhs;
  std::string op;
  std::unique_ptr<ASTNode> rhs;
public:
  InfixExpressionAST(Location loc, std::unique_ptr<ASTNode> lhs,
                     std::string op, std::unique_ptr<ASTNode> rhs)
    : ASTNode(Node_InfixExpression, std::move(loc)),
      lhs(std::move(lhs)), op(std::move(op)), rhs(std::move(rhs)) {}
  
  const ASTNode* getLHS() const { return lhs.get(); }
  const std::string& getOperator() const { return op; }
  const ASTNode* getRHS() const { return rhs.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_InfixExpression;
  }
};

/// Parenthesized expression (e.g., (x + y))
class ParenthesizedExpressionAST : public ASTNode {
  std::unique_ptr<ASTNode> expression;
public:
  ParenthesizedExpressionAST(Location loc, std::unique_ptr<ASTNode> expression)
    : ASTNode(Node_ParenthesizedExpression, std::move(loc)),
      expression(std::move(expression)) {}
  
  const ASTNode* getExpression() const { return expression.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ParenthesizedExpression;
  }
};

/// Match case (e.g., A -> 1, B x -> x + 1)
class MatchCaseAST : public ASTNode {
  std::unique_ptr<ASTNode> pattern;
  std::unique_ptr<ASTNode> expression;
public:
  MatchCaseAST(Location loc, std::unique_ptr<ASTNode> pattern,
               std::unique_ptr<ASTNode> expression)
    : ASTNode(Node_MatchCase, std::move(loc)),
      pattern(std::move(pattern)),
      expression(std::move(expression)) {}
  
  const ASTNode* getPattern() const { return pattern.get(); }
  const ASTNode* getExpression() const { return expression.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_MatchCase;
  }
};

/// Match expression (e.g., match x with | A -> 1 | B i -> i + 1)
class MatchExpressionAST : public ASTNode {
  std::unique_ptr<ASTNode> value;
  std::vector<std::unique_ptr<MatchCaseAST>> cases;
public:
  MatchExpressionAST(Location loc, std::unique_ptr<ASTNode> value,
                    std::vector<std::unique_ptr<MatchCaseAST>> cases)
    : ASTNode(Node_MatchExpression, std::move(loc)),
      value(std::move(value)), cases(std::move(cases)) {}
  
  const ASTNode* getValue() const { return value.get(); }
  const std::vector<std::unique_ptr<MatchCaseAST>>& getCases() const { return cases; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_MatchExpression;
  }
};

/// Value pattern (e.g., x, _)
class ValuePatternAST : public ASTNode {
  std::string name;
public:
  ValuePatternAST(Location loc, std::string name)
    : ASTNode(Node_ValuePattern, std::move(loc)), name(std::move(name)) {}
  
  const std::string& getName() const { return name; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ValuePattern;
  }
};

/// Constructor pattern (e.g., Some x, None)
class ConstructorPatternAST : public ASTNode {
  std::unique_ptr<ConstructorPathAST> constructor;
  std::vector<std::unique_ptr<ASTNode>> arguments;
public:
  ConstructorPatternAST(Location loc, std::unique_ptr<ConstructorPathAST> constructor,
                        std::vector<std::unique_ptr<ASTNode>> arguments)
    : ASTNode(Node_ConstructorPattern, std::move(loc)),
      constructor(std::move(constructor)),
      arguments(std::move(arguments)) {}
  
  const ConstructorPathAST* getConstructor() const { return constructor.get(); }
  const std::vector<std::unique_ptr<ASTNode>>& getArguments() const { return arguments; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ConstructorPattern;
  }
};

/// Typed pattern (e.g., (x : int))
class TypedPatternAST : public ASTNode {
  std::unique_ptr<ASTNode> pattern;
  std::unique_ptr<TypeConstructorPathAST> type;
public:
  TypedPatternAST(Location loc, std::unique_ptr<ASTNode> pattern,
                 std::unique_ptr<TypeConstructorPathAST> type)
    : ASTNode(Node_TypedPattern, std::move(loc)),
      pattern(std::move(pattern)), type(std::move(type)) {}
  
  const ASTNode* getPattern() const { return pattern.get(); }
  const TypeConstructorPathAST* getType() const { return type.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_TypedPattern;
  }
};

/// Constructor declaration (e.g., A, B of int)
class ConstructorDeclarationAST : public ASTNode {
  std::string name;
  std::unique_ptr<TypeConstructorPathAST> ofType;
public:
  ConstructorDeclarationAST(Location loc, std::string name,
                           std::unique_ptr<TypeConstructorPathAST> ofType = nullptr)
    : ASTNode(Node_ConstructorDeclaration, std::move(loc)),
      name(std::move(name)), ofType(std::move(ofType)) {}
  
  const std::string& getName() const { return name; }
  const TypeConstructorPathAST* getOfType() const { return ofType.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ConstructorDeclaration;
  }
};

/// Variant declaration (e.g., A | B of int)
class VariantDeclarationAST : public ASTNode {
  std::vector<std::unique_ptr<ConstructorDeclarationAST>> constructors;
public:
  VariantDeclarationAST(Location loc, 
                       std::vector<std::unique_ptr<ConstructorDeclarationAST>> constructors)
    : ASTNode(Node_VariantDeclaration, std::move(loc)),
      constructors(std::move(constructors)) {}
  
  const std::vector<std::unique_ptr<ConstructorDeclarationAST>>& getConstructors() const {
    return constructors;
  }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_VariantDeclaration;
  }
};

/// Type binding (e.g., shape = A | B of int)
class TypeBindingAST : public ASTNode {
  std::string name;
  std::unique_ptr<ASTNode> definition; // Could be VariantDeclarationAST or other type definitions
public:
  TypeBindingAST(Location loc, std::string name, std::unique_ptr<ASTNode> definition)
    : ASTNode(Node_TypeBinding, std::move(loc)),
      name(std::move(name)), definition(std::move(definition)) {}
  
  const std::string& getName() const { return name; }
  const ASTNode* getDefinition() const { return definition.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_TypeBinding;
  }
};

/// Type definition (e.g., type shape = A | B of int)
class TypeDefinitionAST : public ASTNode {
  std::vector<std::unique_ptr<TypeBindingAST>> bindings;
public:
  TypeDefinitionAST(Location loc, std::vector<std::unique_ptr<TypeBindingAST>> bindings)
    : ASTNode(Node_TypeDefinition, std::move(loc)), bindings(std::move(bindings)) {}
  
  const std::vector<std::unique_ptr<TypeBindingAST>>& getBindings() const { return bindings; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_TypeDefinition;
  }
};

/// Let binding (e.g., let f x = x + 1)
class LetBindingAST : public ASTNode {
  std::string name;
  std::vector<std::unique_ptr<ASTNode>> parameters;
  std::unique_ptr<TypeConstructorPathAST> returnType;
  std::unique_ptr<ASTNode> body;
public:
  LetBindingAST(Location loc, std::string name,
               std::vector<std::unique_ptr<ASTNode>> parameters,
               std::unique_ptr<TypeConstructorPathAST> returnType,
               std::unique_ptr<ASTNode> body)
    : ASTNode(Node_LetBinding, std::move(loc)),
      name(std::move(name)),
      parameters(std::move(parameters)),
      returnType(std::move(returnType)),
      body(std::move(body)) {}
  
  const std::string& getName() const { return name; }
  const std::vector<std::unique_ptr<ASTNode>>& getParameters() const { return parameters; }
  const TypeConstructorPathAST* getReturnType() const { return returnType.get(); }
  const ASTNode* getBody() const { return body.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_LetBinding;
  }
};

/// Value definition (e.g., let x = 1, let f x = x + 1)
class ValueDefinitionAST : public ASTNode {
  std::vector<std::unique_ptr<LetBindingAST>> bindings;
public:
  ValueDefinitionAST(Location loc, std::vector<std::unique_ptr<LetBindingAST>> bindings)
    : ASTNode(Node_ValueDefinition, std::move(loc)), bindings(std::move(bindings)) {}
  
  const std::vector<std::unique_ptr<LetBindingAST>>& getBindings() const { return bindings; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ValueDefinition;
  }
};

/// Expression item (e.g., print_int (1 + 2);;)
class ExpressionItemAST : public ASTNode {
  std::unique_ptr<ASTNode> expression;
public:
  ExpressionItemAST(Location loc, std::unique_ptr<ASTNode> expression)
    : ASTNode(Node_ExpressionItem, std::move(loc)),
      expression(std::move(expression)) {}
  
  const ASTNode* getExpression() const { return expression.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ExpressionItem;
  }
};

/// Compilation unit - Top level container for all declarations
class CompilationUnitAST : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> items;
public:
  CompilationUnitAST(Location loc, std::vector<std::unique_ptr<ASTNode>> items)
    : ASTNode(Node_CompilationUnit, std::move(loc)), items(std::move(items)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getItems() const { return items; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_CompilationUnit;
  }
};

} // namespace ocamlc2

