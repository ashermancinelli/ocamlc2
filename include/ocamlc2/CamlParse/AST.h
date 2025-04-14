#pragma once

#include <memory>
#include <string>
#include <vector>
#include <variant>
#include <optional>
#include "llvm/Support/raw_ostream.h"

namespace ocamlc2 {
inline namespace CamlParse {

class Location {
public:
  Location(int startLine, int startCol, int endLine, int endCol)
    : startLine(startLine), startCol(startCol), endLine(endLine), endCol(endCol) {}

  int getStartLine() const { return startLine; }
  int getStartCol() const { return startCol; }
  int getEndLine() const { return endLine; }
  int getEndCol() const { return endCol; }

private:
  int startLine, startCol, endLine, endCol;
};

class ASTNode;
llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const ASTNode& node);

// OCaml flag types
enum class RecFlag {
  Nonrecursive,
  Recursive
};

enum class DirectionFlag {
  Upto,
  Downto
};

enum class PrivateFlag {
  Private,
  Public
};

enum class MutableFlag {
  Immutable,
  Mutable
};

enum class VirtualFlag {
  Virtual,
  Concrete
};

enum class OverrideFlag {
  Override,
  Fresh
};

enum class ClosedFlag {
  Closed,
  Open
};

// Argument label types
enum class ArgLabel {
  Nolabel,
  Labelled,  // label:T -> ...
  Optional   // ?label:T -> ...
};

// Type parameter variance and injectivity
enum class Variance {
  Covariant,
  Contravariant,
  NoVariance,
  Bivariant
};

enum class Injectivity {
  Injective,
  NoInjectivity
};

// Base class for all AST nodes
class ASTNode {
public:
  enum ASTNodeKind {
    // Structure items
    Node_Structure_Item,
    // Patterns
    Node_Pattern,
    Node_Pattern_Variable,
    Node_Pattern_Constant,
    Node_Pattern_Tuple,
    Node_Pattern_Construct,
    Node_Pattern_Any,
    // Expressions
    Node_Expression,
    Node_Expression_Constant,
    Node_Expression_Variable,
    Node_Expression_Let,
    Node_Expression_Function,
    Node_Expression_Apply,
    Node_Expression_Match,
    Node_Expression_Ifthenelse,
    Node_Expression_Sequence,
    Node_Expression_Construct,
    Node_Expression_Tuple,
    Node_Expression_Array,
    Node_Expression_For,
    Node_Expression_While,
    // Constants
    Node_Constant_Int,
    Node_Constant_Char,
    Node_Constant_Float,
    Node_Constant_String,
    Node_Constant_Int32,
    Node_Constant_Int64,
    Node_Constant_Nativeint,
    // Types
    Node_Core_Type,
    Node_Type_Constr,
    Node_Type_Var,
    Node_Type_Arrow,
    Node_Type_Tuple,
    Node_Type_Poly,
    // Definitions
    Node_Value_Definition,
    Node_Type_Declaration,
    Node_Match_Case,
    // Parameters for functions
    Node_Parameter,
    // Structure bodies
    Node_Structure,
    // Compilation unit
    Node_Compilation_Unit,
  };

  ASTNode(ASTNodeKind kind, Location loc) : kind(kind), loc(loc) {}
  virtual ~ASTNode() = default;

  ASTNodeKind getKind() const { return kind; }
  const Location& getLoc() const { return loc; }

  static const char* getName(ASTNodeKind kind);

private:
  ASTNodeKind kind;
  Location loc;
};

// Constant node types
class ConstantAST : public ASTNode {
public:
  enum ConstantKind {
    Const_Int,
    Const_Char,
    Const_Float,
    Const_String,
    Const_Int32,
    Const_Int64,
    Const_Nativeint
  };

  ConstantAST(ASTNodeKind kind, Location loc, ConstantKind constKind)
    : ASTNode(kind, std::move(loc)), constKind(constKind) {}

  ConstantKind getConstantKind() const { return constKind; }

private:
  ConstantKind constKind;
};

class IntConstantAST : public ConstantAST {
  int64_t value;
  std::optional<std::string> suffix;

public:
  IntConstantAST(Location loc, int64_t value, std::optional<std::string> suffix = std::nullopt)
    : ConstantAST(Node_Constant_Int, std::move(loc), Const_Int), value(value), suffix(suffix) {}

  int64_t getValue() const { return value; }
  const std::optional<std::string>& getSuffix() const { return suffix; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_Int;
  }
};

class CharConstantAST : public ConstantAST {
  char value;

public:
  CharConstantAST(Location loc, char value)
    : ConstantAST(Node_Constant_Char, std::move(loc), Const_Char), value(value) {}

  char getValue() const { return value; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_Char;
  }
};

class FloatConstantAST : public ConstantAST {
  double value;
  std::optional<std::string> suffix;

public:
  FloatConstantAST(Location loc, double value, std::optional<std::string> suffix = std::nullopt)
    : ConstantAST(Node_Constant_Float, std::move(loc), Const_Float), value(value), suffix(suffix) {}

  double getValue() const { return value; }
  const std::optional<std::string>& getSuffix() const { return suffix; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_Float;
  }
};

class StringConstantAST : public ConstantAST {
  std::string value;
  std::optional<std::string> delimiter;

public:
  StringConstantAST(Location loc, std::string value, std::optional<std::string> delimiter = std::nullopt)
    : ConstantAST(Node_Constant_String, std::move(loc), Const_String), value(std::move(value)), delimiter(delimiter) {}

  const std::string& getValue() const { return value; }
  const std::optional<std::string>& getDelimiter() const { return delimiter; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_String;
  }
};

class Int32ConstantAST : public ConstantAST {
  int32_t value;

public:
  Int32ConstantAST(Location loc, int32_t value)
    : ConstantAST(Node_Constant_Int32, std::move(loc), Const_Int32), value(value) {}

  int32_t getValue() const { return value; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_Int32;
  }
};

class Int64ConstantAST : public ConstantAST {
  int64_t value;

public:
  Int64ConstantAST(Location loc, int64_t value)
    : ConstantAST(Node_Constant_Int64, std::move(loc), Const_Int64), value(value) {}

  int64_t getValue() const { return value; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_Int64;
  }
};

class NativeintConstantAST : public ConstantAST {
  int64_t value; // Using int64_t as a representation for nativeint in C++

public:
  NativeintConstantAST(Location loc, int64_t value)
    : ConstantAST(Node_Constant_Nativeint, std::move(loc), Const_Nativeint), value(value) {}

  int64_t getValue() const { return value; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Constant_Nativeint;
  }
};

// Pattern classes
class PatternAST : public ASTNode {
public:
  PatternAST(ASTNodeKind kind, Location loc)
    : ASTNode(kind, std::move(loc)) {}

  static bool classof(const ASTNode* node) {
    return node->getKind() >= Node_Pattern && 
           node->getKind() <= Node_Pattern_Any;
  }
};

class PatternVariableAST : public PatternAST {
  std::string name;

public:
  PatternVariableAST(Location loc, std::string name)
    : PatternAST(Node_Pattern_Variable, std::move(loc)), name(std::move(name)) {}

  const std::string& getName() const { return name; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Pattern_Variable;
  }
};

class PatternConstantAST : public PatternAST {
  std::unique_ptr<ConstantAST> constant;

public:
  PatternConstantAST(Location loc, std::unique_ptr<ConstantAST> constant)
    : PatternAST(Node_Pattern_Constant, std::move(loc)), constant(std::move(constant)) {}

  const ConstantAST* getConstant() const { return constant.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Pattern_Constant;
  }
};

class PatternTupleAST : public PatternAST {
  std::vector<std::unique_ptr<PatternAST>> elements;

public:
  PatternTupleAST(Location loc, std::vector<std::unique_ptr<PatternAST>> elements)
    : PatternAST(Node_Pattern_Tuple, std::move(loc)), elements(std::move(elements)) {}

  const std::vector<std::unique_ptr<PatternAST>>& getElements() const { return elements; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Pattern_Tuple;
  }
};

class PatternConstructAST : public PatternAST {
  std::string constructor;
  std::optional<std::unique_ptr<PatternAST>> argument;

public:
  PatternConstructAST(Location loc, std::string constructor, 
                     std::optional<std::unique_ptr<PatternAST>> argument = std::nullopt)
    : PatternAST(Node_Pattern_Construct, std::move(loc)), 
      constructor(std::move(constructor)), argument(std::move(argument)) {}

  const std::string& getConstructor() const { return constructor; }
  const std::optional<std::unique_ptr<PatternAST>>& getArgument() const { return argument; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Pattern_Construct;
  }
};

class PatternAnyAST : public PatternAST {
public:
  PatternAnyAST(Location loc)
    : PatternAST(Node_Pattern_Any, std::move(loc)) {}

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Pattern_Any;
  }
};

// Expression classes
class ExpressionAST : public ASTNode {
public:
  ExpressionAST(ASTNodeKind kind, Location loc)
    : ASTNode(kind, std::move(loc)) {}

  static bool classof(const ASTNode* node) {
    return node->getKind() >= Node_Expression && 
           node->getKind() <= Node_Expression_While;
  }
};

class ExpressionConstantAST : public ExpressionAST {
  std::unique_ptr<ConstantAST> constant;

public:
  ExpressionConstantAST(Location loc, std::unique_ptr<ConstantAST> constant)
    : ExpressionAST(Node_Expression_Constant, std::move(loc)), 
      constant(std::move(constant)) {}

  const ConstantAST* getConstant() const { return constant.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Constant;
  }
};

class ExpressionVariableAST : public ExpressionAST {
  std::string name;

public:
  ExpressionVariableAST(Location loc, std::string name)
    : ExpressionAST(Node_Expression_Variable, std::move(loc)), 
      name(std::move(name)) {}

  const std::string& getName() const { return name; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Variable;
  }
};

class ParameterAST : public ASTNode {
  ArgLabel label;
  std::string labelName; // For Labelled and Optional
  std::unique_ptr<PatternAST> pattern;
  std::optional<std::unique_ptr<ASTNode>> defaultValue;

public:
  ParameterAST(Location loc, ArgLabel label, std::string labelName, 
              std::unique_ptr<PatternAST> pattern,
              std::optional<std::unique_ptr<ASTNode>> defaultValue = std::nullopt)
    : ASTNode(Node_Parameter, std::move(loc)), 
      label(label), labelName(std::move(labelName)), pattern(std::move(pattern)), 
      defaultValue(std::move(defaultValue)) {}

  ArgLabel getLabel() const { return label; }
  const std::string& getLabelName() const { return labelName; }
  const PatternAST* getPattern() const { return pattern.get(); }
  const std::optional<std::unique_ptr<ASTNode>>& getDefaultValue() const { return defaultValue; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Parameter;
  }
};

class ExpressionFunctionAST : public ExpressionAST {
  std::vector<std::unique_ptr<ParameterAST>> parameters;
  std::optional<std::unique_ptr<ASTNode>> returnType;
  std::unique_ptr<ExpressionAST> body;

public:
  ExpressionFunctionAST(Location loc,
                       std::vector<std::unique_ptr<ParameterAST>> parameters,
                       std::unique_ptr<ExpressionAST> body,
                       std::optional<std::unique_ptr<ASTNode>> returnType = std::nullopt)
    : ExpressionAST(Node_Expression_Function, std::move(loc)),
      parameters(std::move(parameters)), returnType(std::move(returnType)),
      body(std::move(body)) {}

  const std::vector<std::unique_ptr<ParameterAST>>& getParameters() const { return parameters; }
  const std::optional<std::unique_ptr<ASTNode>>& getReturnType() const { return returnType; }
  const ExpressionAST* getBody() const { return body.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Function;
  }
};

class ValueDefinitionAST : public ASTNode {
  RecFlag recursiveFlag;
  std::unique_ptr<PatternAST> pattern;
  std::unique_ptr<ExpressionAST> expression;

public:
  ValueDefinitionAST(Location loc, RecFlag recursiveFlag, 
                    std::unique_ptr<PatternAST> pattern,
                    std::unique_ptr<ExpressionAST> expression)
    : ASTNode(Node_Value_Definition, std::move(loc)),
      recursiveFlag(recursiveFlag), pattern(std::move(pattern)),
      expression(std::move(expression)) {}

  RecFlag getRecursiveFlag() const { return recursiveFlag; }
  const PatternAST* getPattern() const { return pattern.get(); }
  const ExpressionAST* getExpression() const { return expression.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Value_Definition;
  }
};

class ExpressionLetAST : public ExpressionAST {
  RecFlag recursiveFlag;
  std::vector<std::unique_ptr<ValueDefinitionAST>> definitions;
  std::unique_ptr<ExpressionAST> body;

public:
  ExpressionLetAST(Location loc, RecFlag recursiveFlag,
                  std::vector<std::unique_ptr<ValueDefinitionAST>> definitions,
                  std::unique_ptr<ExpressionAST> body)
    : ExpressionAST(Node_Expression_Let, std::move(loc)),
      recursiveFlag(recursiveFlag), definitions(std::move(definitions)),
      body(std::move(body)) {}

  RecFlag getRecursiveFlag() const { return recursiveFlag; }
  const std::vector<std::unique_ptr<ValueDefinitionAST>>& getDefinitions() const { return definitions; }
  const ExpressionAST* getBody() const { return body.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Let;
  }
};

class ExpressionApplyAST : public ExpressionAST {
  std::unique_ptr<ExpressionAST> function;
  std::vector<std::pair<ArgLabel, std::unique_ptr<ExpressionAST>>> arguments;

public:
  ExpressionApplyAST(Location loc, std::unique_ptr<ExpressionAST> function,
                    std::vector<std::pair<ArgLabel, std::unique_ptr<ExpressionAST>>> arguments)
    : ExpressionAST(Node_Expression_Apply, std::move(loc)),
      function(std::move(function)), arguments(std::move(arguments)) {}

  const ExpressionAST* getFunction() const { return function.get(); }
  const std::vector<std::pair<ArgLabel, std::unique_ptr<ExpressionAST>>>& getArguments() const { return arguments; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Apply;
  }
};

class MatchCaseAST : public ASTNode {
  std::unique_ptr<PatternAST> pattern;
  std::optional<std::unique_ptr<ExpressionAST>> guard;
  std::unique_ptr<ExpressionAST> expression;

public:
  MatchCaseAST(Location loc, std::unique_ptr<PatternAST> pattern,
               std::unique_ptr<ExpressionAST> expression,
               std::optional<std::unique_ptr<ExpressionAST>> guard = std::nullopt)
    : ASTNode(Node_Match_Case, std::move(loc)),
      pattern(std::move(pattern)), guard(std::move(guard)), 
      expression(std::move(expression)) {}

  const PatternAST* getPattern() const { return pattern.get(); }
  const std::optional<std::unique_ptr<ExpressionAST>>& getGuard() const { return guard; }
  const ExpressionAST* getExpression() const { return expression.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Match_Case;
  }
};

class ExpressionMatchAST : public ExpressionAST {
  std::unique_ptr<ExpressionAST> expression;
  std::vector<std::unique_ptr<MatchCaseAST>> cases;

public:
  ExpressionMatchAST(Location loc, std::unique_ptr<ExpressionAST> expression,
                    std::vector<std::unique_ptr<MatchCaseAST>> cases)
    : ExpressionAST(Node_Expression_Match, std::move(loc)),
      expression(std::move(expression)), cases(std::move(cases)) {}

  const ExpressionAST* getExpression() const { return expression.get(); }
  const std::vector<std::unique_ptr<MatchCaseAST>>& getCases() const { return cases; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Match;
  }
};

class ExpressionIfthenelseAST : public ExpressionAST {
  std::unique_ptr<ExpressionAST> condition;
  std::unique_ptr<ExpressionAST> thenExpr;
  std::optional<std::unique_ptr<ExpressionAST>> elseExpr;

public:
  ExpressionIfthenelseAST(Location loc, std::unique_ptr<ExpressionAST> condition,
                         std::unique_ptr<ExpressionAST> thenExpr,
                         std::optional<std::unique_ptr<ExpressionAST>> elseExpr = std::nullopt)
    : ExpressionAST(Node_Expression_Ifthenelse, std::move(loc)),
      condition(std::move(condition)), thenExpr(std::move(thenExpr)),
      elseExpr(std::move(elseExpr)) {}

  const ExpressionAST* getCondition() const { return condition.get(); }
  const ExpressionAST* getThenExpr() const { return thenExpr.get(); }
  const std::optional<std::unique_ptr<ExpressionAST>>& getElseExpr() const { return elseExpr; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Ifthenelse;
  }
};

class ExpressionForAST : public ExpressionAST {
  std::unique_ptr<PatternAST> pattern;
  std::unique_ptr<ExpressionAST> startExpr;
  std::unique_ptr<ExpressionAST> endExpr;
  DirectionFlag direction;
  std::unique_ptr<ExpressionAST> body;

public:
  ExpressionForAST(Location loc, 
                  std::unique_ptr<PatternAST> pattern,
                  std::unique_ptr<ExpressionAST> startExpr,
                  std::unique_ptr<ExpressionAST> endExpr,
                  DirectionFlag direction,
                  std::unique_ptr<ExpressionAST> body)
    : ExpressionAST(Node_Expression_For, std::move(loc)),
      pattern(std::move(pattern)), 
      startExpr(std::move(startExpr)), 
      endExpr(std::move(endExpr)),
      direction(direction),
      body(std::move(body)) {}

  const PatternAST* getPattern() const { return pattern.get(); }
  const ExpressionAST* getStartExpr() const { return startExpr.get(); }
  const ExpressionAST* getEndExpr() const { return endExpr.get(); }
  DirectionFlag getDirection() const { return direction; }
  const ExpressionAST* getBody() const { return body.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_For;
  }
};

class ExpressionWhileAST : public ExpressionAST {
  std::unique_ptr<ExpressionAST> condition;
  std::unique_ptr<ExpressionAST> body;

public:
  ExpressionWhileAST(Location loc, 
                    std::unique_ptr<ExpressionAST> condition,
                    std::unique_ptr<ExpressionAST> body)
    : ExpressionAST(Node_Expression_While, std::move(loc)),
      condition(std::move(condition)), 
      body(std::move(body)) {}

  const ExpressionAST* getCondition() const { return condition.get(); }
  const ExpressionAST* getBody() const { return body.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_While;
  }
};

class ExpressionSequenceAST : public ExpressionAST {
  std::vector<std::unique_ptr<ExpressionAST>> expressions;

public:
  ExpressionSequenceAST(Location loc, std::vector<std::unique_ptr<ExpressionAST>> expressions)
    : ExpressionAST(Node_Expression_Sequence, std::move(loc)),
      expressions(std::move(expressions)) {}

  const std::vector<std::unique_ptr<ExpressionAST>>& getExpressions() const { return expressions; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Sequence;
  }
};

class ExpressionConstructAST : public ExpressionAST {
  std::string constructor;
  std::optional<std::unique_ptr<ExpressionAST>> argument;

public:
  ExpressionConstructAST(Location loc, std::string constructor,
                        std::optional<std::unique_ptr<ExpressionAST>> argument = std::nullopt)
    : ExpressionAST(Node_Expression_Construct, std::move(loc)),
      constructor(std::move(constructor)), argument(std::move(argument)) {}

  const std::string& getConstructor() const { return constructor; }
  const std::optional<std::unique_ptr<ExpressionAST>>& getArgument() const { return argument; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Construct;
  }
};

class ExpressionTupleAST : public ExpressionAST {
  std::vector<std::unique_ptr<ExpressionAST>> elements;

public:
  ExpressionTupleAST(Location loc, std::vector<std::unique_ptr<ExpressionAST>> elements)
    : ExpressionAST(Node_Expression_Tuple, std::move(loc)),
      elements(std::move(elements)) {}

  const std::vector<std::unique_ptr<ExpressionAST>>& getElements() const { return elements; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Tuple;
  }
};

class ExpressionArrayAST : public ExpressionAST {
  std::vector<std::unique_ptr<ExpressionAST>> elements;

public:
  ExpressionArrayAST(Location loc, std::vector<std::unique_ptr<ExpressionAST>> elements)
    : ExpressionAST(Node_Expression_Array, std::move(loc)),
      elements(std::move(elements)) {}

  const std::vector<std::unique_ptr<ExpressionAST>>& getElements() const { return elements; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Expression_Array;
  }
};

// Core type classes
class CoreTypeAST : public ASTNode {
public:
  CoreTypeAST(ASTNodeKind kind, Location loc)
    : ASTNode(kind, std::move(loc)) {}

  static bool classof(const ASTNode* node) {
    return node->getKind() >= Node_Core_Type && 
           node->getKind() <= Node_Type_Poly;
  }
};

class TypeVarAST : public CoreTypeAST {
  std::string name;

public:
  TypeVarAST(Location loc, std::string name)
    : CoreTypeAST(Node_Type_Var, std::move(loc)), name(std::move(name)) {}

  const std::string& getName() const { return name; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Type_Var;
  }
};

class TypeConstrAST : public CoreTypeAST {
  std::string name;
  std::vector<std::unique_ptr<CoreTypeAST>> arguments;

public:
  TypeConstrAST(Location loc, std::string name, 
               std::vector<std::unique_ptr<CoreTypeAST>> arguments = {})
    : CoreTypeAST(Node_Type_Constr, std::move(loc)),
      name(std::move(name)), arguments(std::move(arguments)) {}

  const std::string& getName() const { return name; }
  const std::vector<std::unique_ptr<CoreTypeAST>>& getArguments() const { return arguments; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Type_Constr;
  }
};

class TypeArrowAST : public CoreTypeAST {
  std::unique_ptr<CoreTypeAST> left;
  std::unique_ptr<CoreTypeAST> right;

public:
  TypeArrowAST(Location loc, std::unique_ptr<CoreTypeAST> left, std::unique_ptr<CoreTypeAST> right)
    : CoreTypeAST(Node_Type_Arrow, std::move(loc)), 
      left(std::move(left)), right(std::move(right)) {}

  const CoreTypeAST* getLeft() const { return left.get(); }
  const CoreTypeAST* getRight() const { return right.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Type_Arrow;
  }
};

class TypeTupleAST : public CoreTypeAST {
  std::vector<std::unique_ptr<CoreTypeAST>> elements;

public:
  TypeTupleAST(Location loc, std::vector<std::unique_ptr<CoreTypeAST>> elements)
    : CoreTypeAST(Node_Type_Tuple, std::move(loc)), elements(std::move(elements)) {}

  const std::vector<std::unique_ptr<CoreTypeAST>>& getElements() const { return elements; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Type_Tuple;
  }
};

class TypePolyAST : public CoreTypeAST {
  std::vector<std::string> variables;
  std::unique_ptr<CoreTypeAST> type;

public:
  TypePolyAST(Location loc, std::vector<std::string> variables, std::unique_ptr<CoreTypeAST> type)
    : CoreTypeAST(Node_Type_Poly, std::move(loc)), 
      variables(std::move(variables)), type(std::move(type)) {}

  const std::vector<std::string>& getVariables() const { return variables; }
  const CoreTypeAST* getType() const { return type.get(); }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Type_Poly;
  }
};

// Type parameter with variance and injectivity
class TypeParameter {
  std::string name;
  Variance variance;
  Injectivity injectivity;

public:
  TypeParameter(std::string name, Variance variance = Variance::NoVariance, 
               Injectivity injectivity = Injectivity::NoInjectivity)
    : name(std::move(name)), variance(variance), injectivity(injectivity) {}

  const std::string& getName() const { return name; }
  Variance getVariance() const { return variance; }
  Injectivity getInjectivity() const { return injectivity; }
};

// Type definition for variant constructors
class TypeConstructorAST {
  std::string name;
  std::vector<std::unique_ptr<CoreTypeAST>> arguments;
  std::optional<std::unique_ptr<CoreTypeAST>> returnType;
  MutableFlag mutableFlag;
  PrivateFlag privateFlag;

public:
  TypeConstructorAST(std::string name, 
                    std::vector<std::unique_ptr<CoreTypeAST>> arguments = {},
                    std::optional<std::unique_ptr<CoreTypeAST>> returnType = std::nullopt,
                    MutableFlag mutableFlag = MutableFlag::Immutable,
                    PrivateFlag privateFlag = PrivateFlag::Public)
    : name(std::move(name)), arguments(std::move(arguments)),
      returnType(std::move(returnType)), mutableFlag(mutableFlag), privateFlag(privateFlag) {}

  const std::string& getName() const { return name; }
  const std::vector<std::unique_ptr<CoreTypeAST>>& getArguments() const { return arguments; }
  const std::optional<std::unique_ptr<CoreTypeAST>>& getReturnType() const { return returnType; }
  MutableFlag getMutableFlag() const { return mutableFlag; }
  PrivateFlag getPrivateFlag() const { return privateFlag; }
};

// Type declaration
class TypeDeclarationAST : public ASTNode {
  RecFlag recursiveFlag;
  std::string name;
  std::vector<TypeParameter> parameters;
  std::vector<TypeConstructorAST> constructors;
  PrivateFlag privateFlag;
  std::optional<std::unique_ptr<CoreTypeAST>> manifest;

public:
  TypeDeclarationAST(Location loc, RecFlag recursiveFlag, std::string name,
                    std::vector<TypeParameter> parameters,
                    std::vector<TypeConstructorAST> constructors,
                    PrivateFlag privateFlag = PrivateFlag::Public,
                    std::optional<std::unique_ptr<CoreTypeAST>> manifest = std::nullopt)
    : ASTNode(Node_Type_Declaration, std::move(loc)),
      recursiveFlag(recursiveFlag), name(std::move(name)),
      parameters(std::move(parameters)), constructors(std::move(constructors)),
      privateFlag(privateFlag), manifest(std::move(manifest)) {}

  RecFlag getRecursiveFlag() const { return recursiveFlag; }
  const std::string& getName() const { return name; }
  const std::vector<TypeParameter>& getParameters() const { return parameters; }
  const std::vector<TypeConstructorAST>& getConstructors() const { return constructors; }
  PrivateFlag getPrivateFlag() const { return privateFlag; }
  const std::optional<std::unique_ptr<CoreTypeAST>>& getManifest() const { return manifest; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Type_Declaration;
  }
};

// Structure item - top level declarations
class StructureItemAST : public ASTNode {
public:
  enum StructureItemKind {
    Str_Value,
    Str_Type,
    Str_Module,
    Str_ModuleType,
    Str_Open,
    Str_Class,
    Str_ClassType,
    Str_Include,
    Str_Attribute,
    Str_Extension,
    Str_Exception,
    Str_ExternalValue
  };

  StructureItemAST(Location loc, StructureItemKind structKind)
    : ASTNode(Node_Structure_Item, std::move(loc)), structKind(structKind) {}

  StructureItemKind getStructureItemKind() const { return structKind; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Structure_Item;
  }

private:
  StructureItemKind structKind;
};

class StructureValueAST : public StructureItemAST {
  RecFlag recursiveFlag;
  std::vector<std::unique_ptr<ValueDefinitionAST>> definitions;

public:
  StructureValueAST(Location loc, RecFlag recursiveFlag,
                   std::vector<std::unique_ptr<ValueDefinitionAST>> definitions)
    : StructureItemAST(loc, Str_Value),
      recursiveFlag(recursiveFlag), definitions(std::move(definitions)) {}

  RecFlag getRecursiveFlag() const { return recursiveFlag; }
  const std::vector<std::unique_ptr<ValueDefinitionAST>>& getDefinitions() const { return definitions; }
};

class StructureTypeAST : public StructureItemAST {
  std::vector<std::unique_ptr<TypeDeclarationAST>> declarations;

public:
  StructureTypeAST(Location loc, std::vector<std::unique_ptr<TypeDeclarationAST>> declarations)
    : StructureItemAST(loc, Str_Type),
      declarations(std::move(declarations)) {}

  const std::vector<std::unique_ptr<TypeDeclarationAST>>& getDeclarations() const { return declarations; }
};

// Structure (module) definition
class StructureAST : public ASTNode {
  std::vector<std::unique_ptr<StructureItemAST>> items;

public:
  StructureAST(Location loc, std::vector<std::unique_ptr<StructureItemAST>> items)
    : ASTNode(Node_Structure, std::move(loc)), items(std::move(items)) {}

  const std::vector<std::unique_ptr<StructureItemAST>>& getItems() const { return items; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Structure;
  }
};

class CompilationUnitAST : public ASTNode {
  std::vector<std::unique_ptr<StructureAST>> structures;

public:
  CompilationUnitAST(Location loc, std::vector<std::unique_ptr<StructureAST>> structures)
    : ASTNode(Node_Compilation_Unit, std::move(loc)), structures(std::move(structures)) {}

  const std::vector<std::unique_ptr<StructureAST>>& getStructures() const { return structures; }

  static bool classof(const ASTNode* node) {
    return node->getKind() == Node_Compilation_Unit;
  }
};

} // namespace CamlParse
} // namespace ocamlc2
