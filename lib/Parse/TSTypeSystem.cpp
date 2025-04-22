#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <sstream>

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {
inline namespace ts {
using namespace ::ts;
using namespace llvm;
using std::move;

namespace { 
struct StringArena {
  std::set<std::string> pool;
  llvm::StringRef save(std::string str) {
    auto [it, _] = pool.insert(str);
    return *it;
  }
  llvm::StringRef save(std::string_view str) {
    return save(std::string(str));
  }
  llvm::StringRef save(const char *str) {
    return save(std::string_view(str));
  }
};
static StringArena stringArena;

static constexpr std::string_view pathTypes[] = {
    "value_path",
    "module_path",
    "constructor_path",
    "type_constructor_path",
};
} // namespace

static std::string getPath(llvm::ArrayRef<llvm::StringRef> path) {
  return llvm::join(path, ".");
}

static std::string hashPath(llvm::ArrayRef<llvm::StringRef> path) {
  return llvm::join(path, "MM");
}

static std::string hashPath(std::vector<std::string> path) {
  return llvm::join(path, "MM");
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const Unifier::Env& env) {
  for (auto it = env.TopLevelMap.begin(); it != env.TopLevelMap.end(); ++it) {
    os << it->first << " -> " << *(it->second->getValue()) << '\n';
  }
  return os;
}

void Unifier::pushModuleSearchPath(llvm::ArrayRef<llvm::StringRef> modules) {
  auto path = hashPath(modules);
  DBGS("pushing module search path: " << path << '\n');
  moduleSearchPath.push_back(stringArena.save(path));
}

void Unifier::pushModule(llvm::StringRef module) {
  DBGS("pushing module: " << module << '\n');
  currentModule.push_back(stringArena.save(module.str()));
}

void Unifier::popModuleSearchPath() {
  DBGS("popping module search path: " << moduleSearchPath.back() << '\n');
  moduleSearchPath.pop_back();
}

void Unifier::popModule() {
  DBGS("popping module: " << currentModule.back() << '\n');
  currentModule.pop_back();
}

std::string Unifier::getHashedPath(llvm::ArrayRef<llvm::StringRef> path) {
  if (currentModule.size() > 0) {
    auto currentModulePath = currentModule;
    currentModulePath.insert(currentModulePath.end(), path.begin(), path.end());
    return hashPath(currentModulePath);
  }
  return hashPath(path);
}

TypeExpr* Unifier::declare(llvm::StringRef name, TypeExpr* type) {
  auto str = stringArena.save(getHashedPath({name}));
  DBGS("Declaring: " << str << " as " << *type << '\n');
  if (name == getWildcardType()->getName() or name == getUnitType()->getName()) {
    DBGS("probably declaring a wildcard variable in a constructor pattern or "
         "assigning to unit, skipping\n");
    return type;
  }
  if (env.count(str)) {
    DBGS("WARNING: Type of " << name << " redeclared\n");
  }
  env.insert(str, type);
  return type;
}

TypeExpr* Unifier::declarePath(llvm::ArrayRef<llvm::StringRef> path, TypeExpr* type) {
  auto hashedPath = hashPath(path);
  DBGS("Declaring path: " << getPath(path) << '(' << hashedPath << ')' << " as " << *type << '\n');
  return declare(hashedPath, type);
}

TypeExpr* Unifier::getType(std::vector<std::string> path) {
  auto hashedPath = hashPath(path);
  return getType(stringArena.save(hashedPath));
}
TypeExpr* Unifier::getType(const std::string_view name) {
  return getType(stringArena.save(name));
}

TypeExpr* Unifier::getType(const llvm::StringRef name) {
  DBGS("Getting type: " << name << '\n');
  if (auto *type = env.lookup(name)) {
    DBGS("Found type in default env: " << *type << '\n');
    return clone(type);
  }
  for (auto &path : llvm::reverse(moduleSearchPath)) {
    DBGS("Checking module search path: " << path << '\n');
    auto possiblePath = hashPath(std::vector<std::string>{path.str(), name.str()});
    auto str = stringArena.save(possiblePath);
    DBGS("with name: " << str << '\n');
    if (auto *type = env.lookup(str)) {
      DBGS("Found type " << *type << " in module search path: " << possiblePath << '\n');
      return clone(type);
    }
  }
  DBGS("Type not declared: " << name << '\n');
  assert(false && "Type not declared");
  return nullptr;
}

void Unifier::initializeEnvironment() {
  DBGS("Initializing environment\n");
  pushModuleSearchPath("Stdlib");

  // We need to insert these directly because other type initializations require
  // varargs, wildcard, etc to define themselves.
  for (std::string_view name : {"int", "float", "bool", "string", "unit", "_", "•", "varargs!"}) {
    auto str = stringArena.save(name);
    DBGS("Declaring type operator: " << str << '\n');
    env.insert(str, createTypeOperator(str));
  }

  pushModule("Stdlib");
  for (std::string_view name : {"int", "float", "bool", "string"}) {
    declare(name, createFunction({createTypeVariable(), getType(name)}));
  }

  auto *T_bool = getBoolType();
  auto *T_float = getFloatType();
  auto *T_int = getIntType();
  auto *T_unit = getUnitType();
  auto *T_string = getStringType();
  auto *T1 = createTypeVariable(), *T2 = createTypeVariable();

  declare("sqrt", createFunction({T_float, T_float}));
  declare("print_int", createFunction({T_int, T_unit}));
  declare("print_endline", createFunction({T_string, T_unit}));
  declare("print_string", createFunction({T_string, T_unit}));
  declare("print_int", createFunction({T_int, T_unit}));
  declare("print_float", createFunction({T_float, T_unit}));
  declare("string_of_int", createFunction({T_int, T_string}));
  declare("float_of_int", createFunction({T_int, T_float}));
  declare("int_of_float", createFunction({T_float, T_int}));
  popModule();

  {
    for (auto arithmetic : {"+", "-", "*", "/", "%"}) {
      declare(arithmetic, createFunction({T_int, T_int, T_int}));
      declare(std::string(arithmetic) + ".",
              createFunction({T_float, T_float, T_float}));
    }
    for (auto comparison : {"=", "!=", "<", "<=", ">", ">="}) {
      declare(comparison, createFunction({T1, T1, T_bool}));
    }
  }
  {
    auto *concatLHS = createFunction({T1, T2});
    auto *concatType = createFunction({concatLHS, T1, T2});
    declare("@@", concatType);
  }
  {

    // Builtin constructors
    auto *Optional =
        createTypeOperator("Optional", {(T1 = createTypeVariable())});
    declare("None", Optional);
    declare("Some", createFunction({T1, Optional}));
  }
  declarePath({"String", "concat"}, createFunction({T_string, getListTypeOf(T_string), T_string}));
  {
    detail::ModuleScope ms{*this, "Printf"};
    declare("printf", createFunction({T_string, getType(std::string_view("varargs!")), T_unit}));
  }

  declarePath({"Float", "pi"}, T_float);

  auto *List = createTypeOperator("List", {(T1 = createTypeVariable())});
  declare("Nil", createFunction({List}));
  declare("Cons", createFunction({T1, List, List}));
  declarePath({"Array", "length"}, createFunction({List, T_int}));
  {
    detail::ModuleScope ms{*this, "List"};
    declare("map", createFunction({createFunction({T1, T2}), getListTypeOf(T1),
                                        getListTypeOf(T2)}));
    declare("fold_left",
            createFunction({createFunction({T1, T2, T1}), T1, List, T2}));
    declare("fold_right", getType(std::string_view("fold_left")));
  }
}

llvm::raw_ostream& Unifier::show(ts::Cursor cursor, bool showUnnamed) {
  auto showTypes = [this](llvm::raw_ostream &os, ts::Node node) {
    if (auto *te = nodeToType.lookup(node.getID())) {
      os << ANSIColors::magenta() << " " << *te << ANSIColors::reset();
    }
  };
  return dump(llvm::errs(), cursor.copy(), source, 0, showUnnamed, showTypes);
}

TypeExpr* Unifier::infer(ts::Node const& ast) {
  return infer(ast.getCursor());
}

TypeExpr* Unifier::setType(Node node, TypeExpr *type) {
  nodeToType[node.getID()] = type;
  return type;
}

TypeExpr* Unifier::infer(ts::Cursor cursor) {
  DBGS("Inferring type for: " << cursor.getCurrentNode().getType() << '\n');
  DBG(show(cursor.copy(), true));
  auto *te = inferType(cursor.copy());
  DBGS("Inferred type:\n");
  setType(cursor.getCurrentNode(), te);
  DBG(show(cursor.copy(), true));
  return te;
}

TypeExpr* Unifier::inferConstructorPattern(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructor_pattern");
  auto *ctorType = [&] {
    auto name = node.getNamedChild(0);
    auto *ctorType = inferConstructorPath(name.getCursor());
    auto *ctorTypeOperator = llvm::dyn_cast<TypeOperator>(ctorType);
    assert(ctorTypeOperator && "Expected constructor type operator");
    return ctorTypeOperator;
  }();
  auto declaredArgs = SmallVector<TypeExpr*>(ctorType->getArgs());
  if (declaredArgs.empty()) {
    // If we get an empty type operator, we're matching against a constructor
    // with no arguments, so we can just return the type operator directly
    // and there's nothing to unify.
    return ctorType;
  }
  // Otherwise, we have a function mapping the variant constructor arguments to the variant type
  // itself, so we drop the return type to get the pattern arguments to match against.
  auto *varaintType = declaredArgs.pop_back_val();
  auto argsType = [&] {
    auto ctorArgumentsPattern = node.getNamedChild(1);
    auto *argsType = infer(ctorArgumentsPattern);
    return argsType;
  }();
  if (declaredArgs.size() == 1) {
    // If we have a single argument, matches will be directly against it and not wrapped
    // in a tuple type.
    if (failed(unify(argsType, declaredArgs[0]))) {
      assert(false && "Failed to unify constructor pattern argument with constructor type operator argument");
      return nullptr;
    }
    return varaintType;
  } else {
    auto args = llvm::cast<TypeOperator>(argsType)->getArgs();
    assert(args.size() == declaredArgs.size() && "Expected constructor pattern to have the same number of "
                      "arguments as the constructor type operator");
    for (auto [arg, declaredArg] : llvm::zip(args, declaredArgs)) {
      if (failed(unify(arg, declaredArg))) {
        assert(false && "Failed to unify constructor pattern argument with "
                        "constructor type operator argument");
        return nullptr;
      }
    }
    return varaintType;
  }
}

TypeExpr* Unifier::inferTuplePattern(ts::Node node) {
  SmallVector<TypeExpr*> types;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    types.push_back(infer(node.getNamedChild(i)));
  }
  return createTuple(types);
}

TypeExpr* Unifier::inferPattern(ts::Node node) {
  static constexpr std::string_view passthroughPatterns[] = {
    "number", "string",
  };
  if (node.getType() == "value_pattern") {
    auto *type = createTypeVariable();
    declare(node, type);
    return type;
  } else if (node.getType() == "constructor_path") {
    return inferConstructorPath(node.getCursor());
  } else if (node.getType() == "parenthesized_pattern") {
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "constructor_pattern") {
    return inferConstructorPattern(node.getCursor());
  } else if (node.getType() == "tuple_pattern") {
    return inferTuplePattern(node);
  } else if (llvm::is_contained(passthroughPatterns, node.getType())) {
    return infer(node);
  }
  show(node.getCursor(), true);
  DBGS("Unknown pattern type: " << node.getType() << '\n');
  assert(false && "Unknown pattern type");
  return nullptr;
}

TypeExpr* Unifier::inferMatchCase(TypeExpr* matcheeType, ts::Node node) {
  // Declared variables are only in scope during the body of the match case.
  detail::Scope scope(this);
  assert(node.getType() == "match_case");
  const auto namedChildren = node.getNumNamedChildren();
  const auto hasGuard = namedChildren == 3;
  const auto bodyNode = hasGuard ? node.getNamedChild(2) : node.getNamedChild(1);
  auto *matchCaseType = infer(node.getNamedChild(0));
  if (failed(unify(matchCaseType, matcheeType))) {
    assert(false && "Failed to unify match case type with matchee type");
    return nullptr;
  }
  if (hasGuard) {
    auto *guardType = infer(node.getNamedChild(1));
    if (failed(unify(guardType, getBoolType()))) {
      assert(false && "Failed to unify guard type with bool type");
      return nullptr;
    }
  }
  auto *bodyType = infer(bodyNode);
  setType(bodyNode, bodyType);
  return bodyType;
}

TypeExpr *Unifier::inferMatchExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "match_expression");
  auto matchee = node.getNamedChild(0);
  const auto namedChildren = node.getNumNamedChildren();
  if (matchee.getType() == "match_case") {
    assert(false && "NYI");
    // If we don't get a match-case, we're implicitly creating
    // a function with the matchee not available as a symbol yet.
    auto *matcheeType = createTypeVariable();
    declare(node, matcheeType);
    SmallVector<TypeExpr*> functionType = {matcheeType};
    unsigned i = 1;
    while (i < namedChildren) {
      auto *type = inferMatchCase(matcheeType, node.getNamedChild(i++));
      // size .gt. 1, because the implicit matchee is the first element of the case types
      if (functionType.size() > 1) {
        if (failed(unify(type, functionType.back()))) {
          assert(false && "Failed to unify case type with previous case type");
          return nullptr;
        }
      }
      functionType.push_back(type);
    }
    return createFunction(functionType);
  } else {
    // Otherwise, we're matching against a value, so we need to infer the type of the
    // matchee.
    auto *matcheeType = infer(matchee);
    TypeExpr *resultType = nullptr;
    unsigned i = 1;
    while (i < namedChildren) {
      auto caseNode = node.getNamedChild(i++);
      auto *type = inferMatchCase(matcheeType, caseNode);
      if (resultType != nullptr && failed(unify(type, resultType))) {
        assert(false && "Failed to unify case type with previous case type");
        return nullptr;
      }
      resultType = type;
    }
    return resultType;
  }
}

static bool isLetBindingRecursive(Cursor ast) {
  DBGS("Checking: " << ast.getCurrentNode().getType() << '\n');
  auto node = ast.getCurrentNode();
  auto maybeRecKeyword = node.getPreviousSibling();
  if (!maybeRecKeyword.isNull() && maybeRecKeyword.getType() == "rec") {
    return true;
  }
  return false;
}

TypeExpr *Unifier::inferLetBindingFunction(Node name, SmallVector<Node> parameters, Node body) {
  DBGS("non-recursive let binding, inferring body type\n");
  auto [returnType, types] = [&]() { 
    detail::Scope scope(this);
    SmallVector<TypeExpr*> types = llvm::map_to_vector(parameters, [&](Node n) -> TypeExpr* {
      if (n.getType() == "unit") {
        return getUnitType();
      }
      auto *tv = createTypeVariable();
      declare(n, tv);
      return tv;
    });
    auto *returnType = infer(body);
    return std::make_pair(returnType, types);
  }();
  types.push_back(returnType);
  auto *funcType = createFunction(types);
  declare(name, funcType);
  return funcType;
}

TypeExpr *Unifier::inferLetBindingRecursiveFunction(Node name, SmallVector<Node> parameters, Node body) {
  DBGS("recursive let binding, declaring function type before body\n");
  auto *tv = createTypeVariable();
  declare(name, tv);
  auto *funcType = [&]() -> TypeExpr* {
    detail::Scope scope(this);
    SmallVector<TypeExpr*> types = llvm::map_to_vector(parameters, [&](Node n) -> TypeExpr* {
      if (n.getType() == "unit") {
        return getUnitType();
      }
      auto *tv = createTypeVariable();
      declare(n, tv);
      return tv;
    });
    auto *bodyType = infer(body);
    types.push_back(bodyType);
    return createFunction(types);
  }();
  if (failed(unify(funcType, tv))) {
    assert(false && "Failed to unify function type with type variable");
    return nullptr;
  }
  return funcType;
}

TypeExpr *Unifier::inferLetBindingValue(Node name, Node body) {
  DBGS("variable let binding, no parameters\n");
  auto *bodyType = infer(body);
  declare(name, bodyType);
  return bodyType;
}


TypeExpr* Unifier::inferLetBinding(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  SmallVector<Node> parameters;
  const bool isRecursive = isLetBindingRecursive(ast.copy());
  auto name = node.getNamedChild(0),
       body = node.getNamedChild(node.getNumNamedChildren() - 1);
  for (unsigned i = 1; i < node.getNumNamedChildren() - 1; ++i) {
    auto n = node.getNamedChild(i);
    assert(n.getType() == "parameter" && "Expected parameter");
    parameters.push_back(n.getNamedChild(0));
  }
  if (parameters.empty()) {
    assert(!isRecursive && "Expected non-recursive let binding for non-function");
    return inferLetBindingValue(name, body);
  } else if (isRecursive) {
    return inferLetBindingRecursiveFunction(name, parameters, body);
  } else {
    return inferLetBindingFunction(name, parameters, body);
  }
}

TypeExpr* Unifier::inferIfExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  auto childCount = node.getNumNamedChildren();
  auto *condition = infer(node.getNamedChild(0));
  if (failed(unify(condition, getBoolType()))) {
    assert(false &&
           "Failed to unify condition of if expression with bool type");
    return nullptr;
  }
  auto *resultType = createTypeVariable();
  switch (childCount) {
    case 3: {
      auto *thenBranch = infer(node.getNamedChild(1));
      auto *elseBranch = infer(node.getNamedChild(2));
      if (failed(unify(thenBranch, resultType))) {
        assert(false && "Failed to unify then branch of if expression with result type");
        return nullptr;
      }
      if (failed(unify(elseBranch, resultType))) {
        assert(false && "Failed to unify else branch of if expression with result type");
        return nullptr;
      }
      return resultType;
    }
    case 2: {
      auto *resultType = createTypeVariable();
      auto *thenBranch = infer(node.getNamedChild(0));
      if (failed(unify(thenBranch, resultType))) {
        assert(false && "Failed to unify then branch of if expression with result type");
        return nullptr;
      }
      if (failed(unify(getUnitType(), resultType))) {
        assert(false && "Failed to unify then branch of if expression with result type");
        return nullptr;
      }
      return resultType;
    }
    default: {
      assert(false && "Expected 2 or 3 children for if expression");
    }
  }
  return nullptr;
}

TypeExpr* Unifier::declare(Node node, TypeExpr* type) {
  return declare(getText(node, source), type);
}

TypeExpr* Unifier::inferForExpression(Cursor ast) {
  assert(ast.gotoFirstChild());
  assert(ast.gotoNextSibling());
  auto id = ast.getCurrentNode();
  declare(id, getIntType());
  assert(ast.gotoNextSibling());
  assert(ast.getCurrentNode().getType() == "=");
  assert(ast.gotoNextSibling());
  auto firstBound = ast.getCurrentNode();
  assert(ast.gotoNextSibling());
  assert(ast.getCurrentNode().getType() == "to" or ast.getCurrentNode().getType() == "downto");
  assert(ast.gotoNextSibling());
  auto secondBound = ast.getCurrentNode();
  assert(ast.gotoNextSibling());
  assert(ast.getCurrentNode().getType() == "do_clause");
  auto body = ast.getCurrentNode();
  assert(!ast.gotoNextSibling());
  if (failed(unify(infer(firstBound), getIntType()))) {
    assert(false && "Failed to unify first bound");
    return nullptr;
  }
  if (failed(unify(infer(secondBound), getIntType()))) {
    assert(false && "Failed to unify second bound");
    return nullptr;
  }
  (void)infer(body);
  return getUnitType();
}

TypeExpr* Unifier::inferCompilationUnit(Cursor ast) {
  TRACE();
  detail::Scope scope(this);
  initializeEnvironment();
  auto *t = getUnitType();
  auto shouldSkip = [](Node node) {
    static constexpr std::string_view shouldSkip[] = {
        "comment",
        ";;",
    };
    return llvm::any_of(shouldSkip, [&](auto s) { return node.getType() == s; });
  };
  if (ast.gotoFirstChild()) {
    do {
      if (!shouldSkip(ast.getCurrentNode())) {
        t = infer(ast.copy());
      }
    } while (ast.gotoNextSibling());
  }
  return t;
}

TypeExpr* Unifier::getType(const char *name) {
  return getType(stringArena.save(name));
}

TypeExpr* Unifier::getType(Node node) {
  DBGS("Getting type for: " << node.getType() << '\n');
  if (llvm::is_contained(pathTypes, node.getType())) {
    return getType(getPathParts(node));
  }
  return getType(stringArena.save(getText(node, source)));
}

std::vector<std::string> Unifier::getPathParts(Node node) {
  std::vector<std::string> pathParts;
  static constexpr std::string_view nameTypes[] = {
    "value_name",
    "module_name",
    "constructor_name",
    "type_constructor",
    "type_variable",
  };
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    auto childType = child.getType();
    if (llvm::is_contained(pathTypes, childType)) {
      auto parts = getPathParts(child);
      pathParts.insert(pathParts.end(), parts.begin(), parts.end());
    } else if (llvm::is_contained(nameTypes, childType)) {
      std::string part{getText(child, source)};
      pathParts.push_back(part);
    } else if (childType == "parenthesized_operator") {
      pathParts.push_back(std::string{getText(child.getNamedChild(0), source)});
    } else {
      assert(false && "Unknown path part type");
    }
  }
  DBGS("Path parts: " << llvm::join(pathParts, "<join>") << '\n');
  return pathParts;
}

TypeExpr* Unifier::inferValuePath(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "value_path" or node.getType() == "module_path");
  auto pathParts = getPathParts(node);
  return getType(pathParts);
}

// TODO: currying
TypeExpr* Unifier::inferApplicationExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "application_expression");
  assert(ast.gotoFirstChild());
  auto id = ast.getCurrentNode();
  SmallVector<TypeExpr*> args;
  while (ast.gotoNextSibling()) {
    args.push_back(infer(ast.getCurrentNode()));
  }
  args.push_back(createTypeVariable());
  auto declaredFuncType = infer(id);
  auto inferredFuncType = createFunction(args);
  if (failed(unify(declaredFuncType, inferredFuncType))) {
    assert(false && "Failed to unify declared and inferred function types");
    return nullptr;
  }
  return inferredFuncType->back();
}

TypeExpr* Unifier::inferConstructorPath(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructor_path");
  auto pathParts = getPathParts(node);
  return getType(pathParts);
}

TypeExpr* Unifier::inferArrayGetExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "array_get_expression");
  auto inferredArrayType = infer(node.getNamedChild(0));
  auto indexType = infer(node.getNamedChild(1));
  if (failed(unify(indexType, getIntType()))) {
    assert(false && "Failed to unify index type with int type");
    return nullptr;
  }
  auto *arrayType = getArrayType();
  if (failed(unify(arrayType, inferredArrayType))) {
    assert(false && "Failed to unify array type with itself");
    return nullptr;
  }
  return arrayType->back();
}

TypeExpr* Unifier::inferInfixExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "infix_expression");
  assert(ast.gotoFirstChild());
  auto lhsType = infer(ast.getCurrentNode());
  assert(ast.gotoNextSibling());
  auto op = ast.getCurrentNode();
  assert(ast.gotoNextSibling());
  auto rhsType = infer(ast.getCurrentNode());
  assert(!ast.gotoNextSibling());
  auto *opType = getType(op);
  auto *funcType = createFunction({lhsType, rhsType, createTypeVariable()});
  if (failed(unify(opType, funcType))) {
    assert(false && "Failed to unify operator type with function type");
    return nullptr;
  }
  return funcType->back();
}

TypeExpr* Unifier::inferGuard(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "guard");
  auto *type = infer(node.getNamedChild(0));
  if (failed(unify(type, getBoolType()))) {
    assert(false && "Failed to unify guard type with bool type");
    return nullptr;
  }
  return getBoolType();
}

TypeExpr* Unifier::inferLetExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "let_expression");
  assert(node.getNumNamedChildren() == 2);
  detail::Scope scope(this);
  infer(node.getNamedChild(0));
  return infer(node.getNamedChild(1));
}

TypeExpr* Unifier::inferListExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "list_expression");
  SmallVector<TypeExpr*> args;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto *type = infer(node.getNamedChild(i));
    if (not args.empty() && failed(unify(type, args.back()))) {
      assert(false && "Failed to unify list element type with previous element type");
      return nullptr;
    }
    args.push_back(type);
  }
  return getListTypeOf(args.back());
}

TypeExpr* Unifier::inferFunctionExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "fun_expression");
  SmallVector<TypeExpr*> types;
  detail::Scope scope(this);
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    if (child.getType() == "parameter") {
      types.push_back(infer(child.getNamedChild(0)));
    } else {
      auto body = child.getNamedChild(0);
      assert(i == node.getNumNamedChildren() - 1 && "Expected body after parameters");
      types.push_back(infer(body));
    }
  }
  return createFunction(types);
}

TypeExpr* Unifier::inferSequenceExpression(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  auto *last = getUnitType();
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    last = infer(node.getNamedChild(i));
  }
  return last;
}

TypeExpr* Unifier::inferModuleDefinition(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_definition");
  return inferModuleBinding(node.getNamedChild(0).getCursor());
}

TypeExpr* Unifier::inferModuleBinding(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "module_binding");
  auto name = node.getChildByFieldName("name");
  detail::ModuleScope ms{*this, getText(name, source)};
  const auto numChildren = node.getNumNamedChildren();
  std::optional<Node> signature;
  std::optional<Node> structure;
  for (unsigned i = 1; i < numChildren; ++i) {
    auto child = node.getNamedChild(i);
    if (child.getType() == "signature") {
      signature = child;
    } else if (child.getType() == "structure") {
      structure = child;
    } else {
      show(child.getCursor(), true);
      assert(false && "Unknown module binding child type");
    }
  }
  if (signature) {
    inferModuleSignature(signature->getCursor());
  }
  if (structure) {
    inferModuleStructure(structure->getCursor());
  }
  return createTypeOperator(
      hashPath(ArrayRef<StringRef>{"Module", getText(name, source)}), {});
}

TypeExpr* Unifier::inferModuleSignature(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "signature");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    infer(child);
  }
  return getUnitType();
}

TypeExpr* Unifier::inferModuleStructure(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "structure");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    infer(child);
  }
  return getUnitType();
}

TypeExpr* Unifier::inferTypeExpression(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "function_type");
  SmallVector<TypeExpr*> args;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    args.push_back(infer(child));
  }
  return createFunction(args);
}

TypeExpr* Unifier::inferValueSpecification(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "value_specification");
  auto name = node.getNamedChild(0);
  auto specification = node.getNamedChild(1);
  if (specification.getType() == "function_type") {
    auto *type = inferTypeExpression(specification.getCursor());
    declare(name, type);
    return type;
  }
  show(ast.copy(), true);
  assert(false && "Unknown value specification type");
  return nullptr;
}

TypeExpr* Unifier::inferRecordDeclaration(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "record_declaration");
  SmallVector<llvm::StringRef> fieldNames;
  SmallVector<TypeExpr*> fieldTypes;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    assert(child.getType() == "field_declaration");
    auto name = child.getNamedChild(0);
    auto type = infer(child.getNamedChild(1));
    auto text = getText(name, source);
    fieldNames.push_back(stringArena.save(text));
    fieldTypes.push_back(type);
  }
  auto recordName = getText(node.getNamedChild(0), source);
  recordTypeFieldOrder[stringArena.save(recordName)] = fieldNames;
  return getRecordType(fieldTypes);
}

TypeExpr* Unifier::inferTypeConstructorPath(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_constructor_path");
  auto pathParts = getPathParts(node);
  return getType(pathParts);
}

TypeExpr* Unifier::inferTypeDefinition(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_definition");
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    auto child = node.getNamedChild(i);
    inferTypeBinding(child.getCursor());
  }
  return getUnitType();
}

TypeExpr* Unifier::inferVariantConstructor(TypeExpr* variantType, Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructor_declaration");
  auto name = node.getNamedChild(0);
  if (node.getNumNamedChildren() == 1) {
    return declare(name, variantType);
  }
  auto parameters = [&] {
    SmallVector<TypeExpr*> types;
    for (unsigned i = 1; i < node.getNumNamedChildren(); ++i) {
      types.push_back(infer(node.getNamedChild(i)));
    }
    return types;
  }();
  parameters.push_back(variantType);
  auto *functionType = createFunction(parameters);
  declare(name, functionType);
  return functionType;
}

TypeExpr* Unifier::inferTypeBinding(Cursor ast) {
  TRACE();
  auto node = ast.getCurrentNode();
  assert(node.getType() == "type_binding");
  unsigned childIndex = 0;
  SmallVector<TypeExpr*> typeVars;
  while (node.getNamedChild(childIndex).getType() == "type_variable") {
    typeVars.push_back(createTypeVariable());
    declare(node.getNamedChild(childIndex), typeVars.back());
    ++childIndex;
  }
  auto name = node.getNamedChild(childIndex++);
  auto *variantType = createTypeOperator(getText(name, source), typeVars);
  declare(name, variantType);
  auto body = node.getNamedChild(childIndex++);
  assert(childIndex == node.getNumNamedChildren());
  if (body.getType() == "variant_declaration") {
    for (unsigned i = 0; i < body.getNumNamedChildren(); ++i) {
      auto child = body.getNamedChild(i);
      inferVariantConstructor(variantType, child.getCursor());
    }
  } else {
    show(body.getCursor(), true);
    assert(false && "Unknown type binding type");
  }
  return getUnitType();
}

TypeExpr* Unifier::inferConstructedType(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "constructed_type");
  SmallVector<TypeExpr*> args;
  for (unsigned i = 0; i < node.getNumNamedChildren(); ++i) {
    args.push_back(infer(node.getNamedChild(i)));
  }
  auto *typeBeingConstructed = args.pop_back_val();
  auto *typeOperator = llvm::dyn_cast<TypeOperator>(typeBeingConstructed);
  assert(typeOperator && "Type being constructed is not a type operator");
  auto typeOperatorName = typeOperator->getName();
  auto *inferredType = createTypeOperator(typeOperatorName, args);
  if (failed(unify(inferredType, typeBeingConstructed))) {
    assert(false && "Failed to unify inferred type with type being constructed");
    return nullptr;
  }
  return inferredType;
}

TypeExpr *Unifier::inferValuePattern(Cursor ast) {
  auto node = ast.getCurrentNode();
  assert(node.getType() == "value_pattern");
  auto text = getText(node, source);
  if (text == "_") {
    return getWildcardType();
  }
  return declare(node, createTypeVariable());
}

TypeExpr* Unifier::inferType(Cursor ast) {
  auto node = ast.getCurrentNode();
  static constexpr std::string_view passthroughTypes[] = {
    "parenthesized_expression",
    "then_clause",
    "else_clause",
    "value_definition",
    "expression_item",
  };
  if (node.getType() == "number") {
    auto text = getText(node, source);
    return text.contains('.') ? getType("float") : getType("int");
  } else if (node.getType() == "float") {
    return getType("float");
  } else if (node.getType() == "unit") {
    return getUnitType();
  } else if (node.getType() == "string") {
    return getType("string");
  } else if (node.getType() == "guard") {
    return inferGuard(std::move(ast));
  } else if (node.getType() == "do_clause") {
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "constructor_path") {
    return inferConstructorPath(std::move(ast));
  } else if (node.getType() == "let_binding") {
    return inferLetBinding(std::move(ast));
  } else if (node.getType() == "let_expression") {
    return inferLetExpression(std::move(ast));
  } else if (node.getType() == "fun_expression") {
    return inferFunctionExpression(std::move(ast));
  } else if (node.getType() == "match_expression") {
    return inferMatchExpression(std::move(ast));
  } else if (node.getType() == "value_path") {
    return inferValuePath(std::move(ast));
  } else if (node.getType() == "for_expression") {
    return inferForExpression(std::move(ast));
  } else if (node.getType() == "infix_expression") {
    return inferInfixExpression(std::move(ast));
  } else if (node.getType() == "if_expression") {
    return inferIfExpression(std::move(ast));
  } else if (node.getType() == "array_get_expression") {
    return inferArrayGetExpression(std::move(ast));
  } else if (node.getType() == "list_expression") {
    return inferListExpression(std::move(ast));
  } else if (llvm::is_contained(passthroughTypes, node.getType())) {
    assert(node.getNumNamedChildren() == 1);
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "compilation_unit") {
    return inferCompilationUnit(std::move(ast));
  } else if (node.getType() == "application_expression") {
    return inferApplicationExpression(std::move(ast));
  } else if (node.getType() == "module_definition") {
    return inferModuleDefinition(std::move(ast));
  } else if (node.getType() == "value_specification") {
    return inferValueSpecification(std::move(ast));
  } else if (node.getType() == "type_constructor_path") {
    return inferTypeConstructorPath(std::move(ast));
  } else if (node.getType() == "record_declaration") {
    return inferRecordDeclaration(std::move(ast));
  } else if (node.getType() == "sequence_expression") {
    return inferSequenceExpression(std::move(ast));
  } else if (node.getType() == "type_definition") {
    return inferTypeDefinition(std::move(ast));
  } else if (node.getType() == "value_pattern") {
    return inferValuePattern(std::move(ast));
  } else if (node.getType() == "constructor_path") {
    return inferConstructorPath(node.getCursor());
  } else if (node.getType() == "parenthesized_pattern") {
    return infer(node.getNamedChild(0));
  } else if (node.getType() == "constructor_pattern") {
    return inferConstructorPattern(std::move(ast));
  } else if (node.getType() == "tuple_pattern") {
    return inferTuplePattern(node);
  } else if (node.getType() == "type_variable") {
    return getType(node);
  } else if (node.getType() == "constructed_type") {
    return inferConstructedType(std::move(ast));
  }
  show(ast.copy(), true);
  llvm::errs() << "Unknown node type: " << node.getType() << '\n';
  assert(false && "Unknown node type");
  return nullptr;
}

// Try to unify two types based on their structure.
// See the paper _Basic Polymorphic Typechecking_ by Luca Cardelli
// for the original unification algorithm.
//
// There are a few main differences used to implement features of OCaml that
// break the original algorithm:
//
// 1. Wildcard unification:
//
//    The original algorithm only unifies two types if they are exactly equal.
//    This is problematic for OCaml's wildcard type '_', which is used to
//    indicate that a type is not important.
//
// 2. Varargs:
//
//    OCaml allows functions to take a variable number of arguments using the
//    varargs type 'varargs!'. This is a special type that is used to indicate
//    that a function can take a variable number of arguments. When two type
//    operators are found and their sizes don't match, we first check if
//    one of the subtypes is varargs. If so, we can unify the types if the
//    other type is a function type.
LogicalResult Unifier::unify(TypeExpr* a, TypeExpr* b) {
  static size_t count = 0;
  auto *wildcard = getWildcardType();
  a = prune(a);
  b = prune(b);
  DBGS("Unifying:" << count++ << ": " << *a << " and " << *b << '\n');
  if (auto *tva = llvm::dyn_cast<TypeVariable>(a)) {
    if (auto *tvb = llvm::dyn_cast<TypeVariable>(b);
        tvb && isConcrete(tva, concreteTypes) &&
        isGeneric(tvb, concreteTypes)) {
      return unify(tvb, tva);
    }
    if (*a != *b) {
      DBGS("Unifying type variable with different type\n");
      if (isSubType(a, b)) {
        assert(false && "Recursive unification");
        return failure();
      }
      tva->instance = b;
    } else {
      DBGS("Unifying type variable with itself, fallthrough\n");
      // fallthrough
    }
  } else if (auto *toa = llvm::dyn_cast<TypeOperator>(a)) {
    if (llvm::isa<TypeVariable>(b)) {
      return unify(b, a);
    } else if (auto *tob = llvm::dyn_cast<TypeOperator>(b)) {

      // Check if we have any varargs types in the arguments - if so, the forms are allowed
      // to be different.
      auto fHasVarargs = [&](TypeExpr* arg) { return isVarargs(arg); };
      const bool hasVarargs = llvm::any_of(toa->getArgs(), fHasVarargs) or
                              llvm::any_of(tob->getArgs(), fHasVarargs);
      DBGS("has varargs: " << hasVarargs << '\n');

      if (toa->getName() == wildcard->getName() or tob->getName() == wildcard->getName()) {
        DBGS("Unifying with wildcard\n");
        return success();
      }

      // Usual type checking - if we don't have a special case, then we need operator types to
      // match in form.
      const bool sizesMatch = toa->getArgs().size() == tob->getArgs().size() or hasVarargs;
      if (toa->getName() != tob->getName() or not sizesMatch) {
        llvm::errs() << "Could not unify types: " << *toa << " and " << *tob << '\n';
        assert(false);
      }

      for (auto [aa, bb] : llvm::zip(toa->getArgs(), tob->getArgs())) {
        if (isVarargs(aa) or isVarargs(bb)) {
          DBGS("Unifying with varargs, unifying return types and giving up\n");
          if (failed(unify(toa->back(), tob->back()))) {
            return failure();
          }
          return success();
        }
        if (isWildcard(aa) or isWildcard(bb)) {
          DBGS("Unifying with wildcard, skipping unification for this element type\n");
          continue;
        }
        if (failed(unify(aa, bb))) {
          return failure();
        }
      }
    }
  }
  return success();
}
TypeExpr *Unifier::clone(TypeExpr *type) {
  llvm::DenseMap<TypeExpr *, TypeExpr *> mapping;
  return clone(type, mapping);
}

TypeExpr *Unifier::clone(TypeExpr *type, llvm::DenseMap<TypeExpr *, TypeExpr *> &mapping) {
  // DBGS("Cloning type: " << *type << '\n');
  type = prune(type);
  // DBGS("recursing on type: " << *type << '\n');
  if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    auto args =
        llvm::to_vector(llvm::map_range(op->getArgs(), [&](TypeExpr *arg) {
          return clone(arg, mapping);
        }));
    // DBGS("cloning type operator: " << op->getName() << '\n');
    return createTypeOperator(op->getName(), args);
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(type)) {
    // DBGS("cloning type variable: " << *tv << '\n');
    if (isGeneric(tv, concreteTypes)) {
      // DBGS("type variable is generic, cloning\n");
      if (mapping.find(tv) == mapping.end()) {
        // DBGS("Didn't find mapping for type variable: "
            //  << *tv << ", creating new one\n");
        mapping[tv] = createTypeVariable();
      }
      // DBGS("returning cloned type variable: " << *mapping[tv] << '\n');
      return mapping[tv];
    }
  }
  // DBGS("cloned type: " << *type << '\n');
  return type;
}

TypeExpr* Unifier::prune(TypeExpr* type) {
  if (auto *tv = llvm::dyn_cast<TypeVariable>(type); tv && tv->instantiated()) {
    tv->instance = prune(tv->instance);
    return tv->instance;
  }
  return type;
}

bool Unifier::isVarargs(TypeExpr* type) {
  if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    return op->getName() == getVarargsType()->getName();
  }
  return false;
}

bool Unifier::isWildcard(TypeExpr* type) {
  if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    return op->getName() == getWildcardType()->getName();
  }
  return false;
}

bool Unifier::isSubType(TypeExpr* a, TypeExpr* b) {
  b = prune(b);
  DBGS("isSubType: " << *a << " and " << *b << '\n');
  if (auto *op = llvm::dyn_cast<TypeOperator>(b)) {
    return isSubTypeOfAny(a, op->getArgs());
  } else if (llvm::isa<TypeVariable>(b)) {
    return *a == *b;
  }
  assert(false && "Unknown type expression");
}

TypeExpr *Unifier::getBoolType() { 
  static TypeExpr *type = getType("bool");
  return type;
}

TypeExpr *Unifier::getFloatType() { 
  static TypeExpr *type = getType("float");
  return type;
}

TypeExpr *Unifier::getIntType() { 
  static TypeExpr *type = getType("int");
  return type;
}

TypeExpr *Unifier::getUnitType() { 
  static TypeExpr *type = getType("unit");
  return type;
}

TypeExpr *Unifier::getStringType() { 
  static TypeExpr *type = getType("string");
  return type;
}

TypeExpr *Unifier::getWildcardType() { 
  static TypeExpr *type = getType("_");
  return type;
}

TypeExpr *Unifier::getVarargsType() { 
  static TypeExpr *type = getType("varargs!");
  return type;
}

} // namespace ts
} // namespace ocamlc2
