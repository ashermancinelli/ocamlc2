#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <variant>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Location.h>
#include <filesystem>
namespace ocamlc2 {

class ASTNode;
struct Location;
struct TypeExpr;
struct Unifier;
std::unique_ptr<ASTNode> parse(const std::string &source, const std::string &filename = "<string>");
std::unique_ptr<ASTNode> parse(const std::filesystem::path &filepath);
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
    Node_String,
    Node_Boolean,
    Node_ValuePath,
    Node_ConstructorPath,
    Node_TypeConstructorPath,
    Node_Application,
    Node_InfixExpression,
    Node_ParenthesizedExpression,
    Node_MatchExpression,
    Node_ForExpression,
    Node_LetExpression,
    Node_IfExpression,
    Node_ListExpression,
    Node_FunExpression,
    Node_UnitExpression,
    Node_SignExpression,
    Node_ArrayGetExpression,
    Node_ArrayExpression,
    Node_SequenceExpression,

    // Patterns
    Node_ValuePattern,
    Node_ConstructorPattern,
    Node_TypedPattern,
    Node_GuardedPattern,
    
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
  TypeExpr *getTypeExpr() const { return typeExpr; }
  friend struct Unifier;

private:
  const ASTNodeKind kind;
  Location location;
  mutable TypeExpr *typeExpr = nullptr;
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

/// String literal expression (e.g., "hello", "world")
class StringExprAST : public ASTNode {
  std::string value;
public:
  StringExprAST(Location loc, std::string value)
    : ASTNode(Node_String, std::move(loc)), value(std::move(value)) {}
  
  const std::string& getValue() const { return value; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_String;
  }
};

/// Boolean literal expression (e.g., true, false)
class BooleanExprAST : public ASTNode {
  bool value;
public:
  BooleanExprAST(Location loc, bool value)
    : ASTNode(Node_Boolean, std::move(loc)), value(value) {}
  
  bool getValue() const { return value; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Boolean;
  }
};

/// Sign expression (e.g., +1, -2)
class SignExpressionAST : public ASTNode {
  std::string op;
  std::unique_ptr<ASTNode> operand;

public:
  SignExpressionAST(Location loc, std::string op,
                    std::unique_ptr<ASTNode> operand)
      : ASTNode(Node_SignExpression, std::move(loc)), op(std::move(op)),
        operand(std::move(operand)) {}

  const std::string &getOperator() const { return op; }
  const ASTNode *getOperand() const { return operand.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Node_SignExpression;
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

/// For expression (e.g., for i = 1 to 10 do ... done)
class ForExpressionAST : public ASTNode {
  std::string loopVar;
  std::unique_ptr<ASTNode> startExpr;
  std::unique_ptr<ASTNode> endExpr;
  std::unique_ptr<ASTNode> body;
  bool isDownto;  // true if it's a downto loop (for i = 10 downto 1), false for upto (for i = 1 to 10)
public:
  ForExpressionAST(Location loc, std::string loopVar,
                  std::unique_ptr<ASTNode> startExpr,
                  std::unique_ptr<ASTNode> endExpr,
                  std::unique_ptr<ASTNode> body,
                  bool isDownto = false)
    : ASTNode(Node_ForExpression, std::move(loc)),
      loopVar(std::move(loopVar)),
      startExpr(std::move(startExpr)),
      endExpr(std::move(endExpr)),
      body(std::move(body)),
      isDownto(isDownto) {}
  
  const std::string& getLoopVar() const { return loopVar; }
  const ASTNode* getStartExpr() const { return startExpr.get(); }
  const ASTNode* getEndExpr() const { return endExpr.get(); }
  const ASTNode* getBody() const { return body.get(); }
  bool getIsDownto() const { return isDownto; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ForExpression;
  }
};

/// If expression (e.g., if x > 0 then y else z)
class IfExpressionAST : public ASTNode {
  std::unique_ptr<ASTNode> condition;
  std::unique_ptr<ASTNode> thenBranch;
  std::unique_ptr<ASTNode> elseBranch;
public:
  IfExpressionAST(Location loc, std::unique_ptr<ASTNode> condition,
                 std::unique_ptr<ASTNode> thenBranch,
                 std::unique_ptr<ASTNode> elseBranch = nullptr)
    : ASTNode(Node_IfExpression, std::move(loc)),
      condition(std::move(condition)),
      thenBranch(std::move(thenBranch)),
      elseBranch(std::move(elseBranch)) {}
  
  const ASTNode* getCondition() const { return condition.get(); }
  const ASTNode* getThenBranch() const { return thenBranch.get(); }
  const ASTNode* getElseBranch() const { return elseBranch.get(); }
  bool hasElseBranch() const { return elseBranch != nullptr; }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_IfExpression;
  }
};

/// List expression (e.g., [1; 2; 3])
class ListExpressionAST : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> elements;
public:
  ListExpressionAST(Location loc, std::vector<std::unique_ptr<ASTNode>> elements)
    : ASTNode(Node_ListExpression, std::move(loc)), elements(std::move(elements)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getElements() const { return elements; }
  size_t getNumElements() const { return elements.size(); }
  ASTNode* getElement(size_t index) const { return elements[index].get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ListExpression;
  }
};

/// Array expression (e.g., [|1; 2; 3|])
class ArrayExpressionAST : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> elements;
public:
  ArrayExpressionAST(Location loc, std::vector<std::unique_ptr<ASTNode>> elements)
    : ASTNode(Node_ArrayExpression, std::move(loc)), elements(std::move(elements)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getElements() const { return elements; }
  size_t getNumElements() const { return elements.size(); }
  ASTNode* getElement(size_t index) const { return elements[index].get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ArrayExpression;
  }
};

/// Function expression (e.g., fun x -> x + 1)
class FunExpressionAST : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> parameters;
  std::unique_ptr<ASTNode> body;
public:
  FunExpressionAST(Location loc, std::vector<std::unique_ptr<ASTNode>> parameters,
                  std::unique_ptr<ASTNode> body)
    : ASTNode(Node_FunExpression, std::move(loc)),
      parameters(std::move(parameters)),
      body(std::move(body)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getParameters() const { return parameters; }
  const ASTNode* getBody() const { return body.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_FunExpression;
  }
};

/// Unit expression (e.g., ())
class UnitExpressionAST : public ASTNode {
public:
  UnitExpressionAST(Location loc)
    : ASTNode(Node_UnitExpression, std::move(loc)) {}
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_UnitExpression;
  }
};

/// Array get expression (e.g., arr.(i))
class ArrayGetExpressionAST : public ASTNode {
  std::unique_ptr<ASTNode> array;
  std::unique_ptr<ASTNode> index;
public:
  ArrayGetExpressionAST(Location loc, std::unique_ptr<ASTNode> array,
                      std::unique_ptr<ASTNode> index)
    : ASTNode(Node_ArrayGetExpression, std::move(loc)),
      array(std::move(array)),
      index(std::move(index)) {}
  
  const ASTNode* getArray() const { return array.get(); }
  const ASTNode* getIndex() const { return index.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_ArrayGetExpression;
  }
};

/// Let expression (e.g., let x = 1 in ...)
class LetExpressionAST : public ASTNode {
  std::unique_ptr<ASTNode> binding;
  std::unique_ptr<ASTNode> body;
public:
  LetExpressionAST(Location loc, std::unique_ptr<ASTNode> binding,
                  std::unique_ptr<ASTNode> body)
    : ASTNode(Node_LetExpression, std::move(loc)),
      binding(std::move(binding)),
      body(std::move(body)) {}
  
  const ASTNode* getBinding() const { return binding.get(); }
  const ASTNode* getBody() const { return body.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_LetExpression;
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

/// Guarded pattern (e.g., x when x > 0)
class GuardedPatternAST : public ASTNode {
  std::unique_ptr<ASTNode> pattern;
  std::unique_ptr<ASTNode> guard;
public:
  GuardedPatternAST(Location loc, std::unique_ptr<ASTNode> pattern,
                   std::unique_ptr<ASTNode> guard)
    : ASTNode(Node_GuardedPattern, std::move(loc)),
      pattern(std::move(pattern)), guard(std::move(guard)) {}
  
  const ASTNode* getPattern() const { return pattern.get(); }
  const ASTNode* getGuard() const { return guard.get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_GuardedPattern;
  }
};

/// Constructor declaration (e.g., A, B of int)
class ConstructorDeclarationAST : public ASTNode {
  std::string name;
  std::vector<std::unique_ptr<TypeConstructorPathAST>> ofTypes;
public:
  ConstructorDeclarationAST(Location loc, std::string name,
                           std::vector<std::unique_ptr<TypeConstructorPathAST>> ofTypes = {})
    : ASTNode(Node_ConstructorDeclaration, std::move(loc)),
      name(std::move(name)), ofTypes(std::move(ofTypes)) {}
  
  const std::string& getName() const { return name; }
  const std::vector<std::unique_ptr<TypeConstructorPathAST>>& getOfTypes() const { return ofTypes; }
  bool hasSingleType() const { return ofTypes.size() == 1; }
  const TypeConstructorPathAST* getOfType() const { return !ofTypes.empty() ? ofTypes[0].get() : nullptr; }
  
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
  bool isRecursive;
public:
  LetBindingAST(Location loc, std::string name,
               std::vector<std::unique_ptr<ASTNode>> parameters,
               std::unique_ptr<TypeConstructorPathAST> returnType,
               std::unique_ptr<ASTNode> body,
               bool isRecursive = false)
    : ASTNode(Node_LetBinding, std::move(loc)),
      name(std::move(name)),
      parameters(std::move(parameters)),
      returnType(std::move(returnType)),
      body(std::move(body)),
      isRecursive(isRecursive) {}
  
  const std::string& getName() const { return name; }
  const std::vector<std::unique_ptr<ASTNode>>& getParameters() const { return parameters; }
  const TypeConstructorPathAST* getReturnType() const { return returnType.get(); }
  const ASTNode* getBody() const { return body.get(); }
  bool getIsRecursive() const { return isRecursive; }
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

/// Sequence expression (e.g., e1; e2; e3)
class SequenceExpressionAST : public ASTNode {
  std::vector<std::unique_ptr<ASTNode>> expressions;
public:
  SequenceExpressionAST(Location loc, std::vector<std::unique_ptr<ASTNode>> expressions)
    : ASTNode(Node_SequenceExpression, std::move(loc)), expressions(std::move(expressions)) {}
  
  const std::vector<std::unique_ptr<ASTNode>>& getExpressions() const { return expressions; }
  size_t getNumExpressions() const { return expressions.size(); }
  ASTNode* getExpression(size_t index) const { return expressions[index].get(); }
  
  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_SequenceExpression;
  }
};

} // namespace ocamlc2

