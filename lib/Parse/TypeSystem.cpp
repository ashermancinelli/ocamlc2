#include "ocamlc2/Parse/TypeSystem.h"
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

namespace { 
struct StringArena {
  std::set<std::string> pool;
  llvm::StringRef save(std::string str) {
    auto [it, _] = pool.insert(str);
    return *it;
  }
};
static StringArena stringArena;
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

TypeVariable::TypeVariable() : TypeExpr(Kind::Variable) {
  static int id = 0;
  this->id = id++;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeVariable& var) {
  if (var.instantiated()) {
    os << "tv:" << *var.instance;
  } else {
    os << var.getName();
  }
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op) {
  auto args = op.getArgs();
  auto name = op.getName().str();
  if (auto pos = name.find("StdlibMM"); pos != std::string::npos) {
    name = name.substr(pos + 8);
  }
  if (args.empty()) {
    return os << name;
  }
  os << '(' << op.getName();
  for (auto *arg : args) {
    os << ' ' << *arg;
  }
  return os << ')';
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeExpr& type) {
  if (auto *to = llvm::dyn_cast<TypeOperator>(&type)) {
    os << *to;
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(&type)) {
    os << *tv;
  }
  return os;
}

bool TypeExpr::operator==(const TypeExpr& other) const {
  if (auto *to = llvm::dyn_cast<TypeOperator>(this)) {
    if (auto *toOther = llvm::dyn_cast<TypeOperator>(&other)) {
      return *to == *toOther;
    }
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(this)) {
    if (auto *tvOther = llvm::dyn_cast<TypeVariable>(&other)) {
      return *tv == *tvOther;
    }
  }
  return false;
}

bool TypeVariable::operator==(const TypeVariable& other) const {
  return id == other.id;
}

TypeExpr* Unifier::infer(const ASTNode* ast) {
  auto *type = inferType(ast);
  ast->typeExpr = type;
  DBGS("\nInferred type: " << *type << " for:\n" << *ast << '\n');
  return type;
}

void Unifier::initializeEnvironment() {
  DBGS("Initializing environment\n");
  pushModuleSearchPath("Stdlib");

  // We need to insert these directly because other type initializations require
  // varargs, wildcard, etc to define themselves.
  for (auto name : {"int", "float", "bool", "string", "unit!", "_", "•", "varargs!"}) {
    auto str = stringArena.save(name);
    DBGS("Declaring type operator: " << str << '\n');
    env.insert(str, createTypeOperator(str));
  }
  auto *T_bool = getBoolType();
  auto *T_float = getFloatType();
  auto *T_int = getIntType();
  auto *T_unit = getUnitType();
  auto *T_string = getStringType();
  auto *T1 = createTypeVariable(), *T2 = createTypeVariable();
  {
    for (auto arithmetic : {"+", "-", "*", "/", "%"}) {
      declare(arithmetic, createFunction({T_int, T_int, T_int}));
      declare(std::string(arithmetic) + ".",
              createFunction({T_float, T_float, T_float}));
    }
    for (auto comparison : {"=", "!=", "<", "<=", ">", ">="}) {
      declare(comparison, createFunction({T1, T1, T_bool}));
    }
    declare("sqrt", createFunction({T_float, T_float}));
    declare("print_int", createFunction({T_int, T_unit}));
    declare("print_endline", createFunction({T_string, T_unit}));
    declare("print_string", createFunction({T_string, T_unit}));
    declare("string_of_int", createFunction({T_int, T_string}));

    // Builtin constructors
    auto *Optional =
        createTypeOperator("Optional", {(T1 = createTypeVariable())});
    declare("None", createFunction({Optional}));
    declare("Some", createFunction({T1, Optional}));
  }
  declarePath({"String", "concat"}, createFunction({T_string, getListOf(T_string), T_string}));
  {
    ModuleScope ms{*this, "Printf"};
    declare("printf", createFunction({T_string, getType("varargs!"), T_unit}));
  }

  declarePath({"Float", "pi"}, T_float);

  auto *List = createTypeOperator("List", {(T1 = createTypeVariable())});
  declare("Nil", createFunction({List}));
  declare("Cons", createFunction({T1, List, List}));
  declarePath({"Array", "length"}, createFunction({List, T_int}));
  {
    ModuleScope ms{*this, "List"};
    declare("map", createFunction({createFunction({T1, T2}), getListOf(T1),
                                        getListOf(T2)}));
    declare("fold_left",
            createFunction({createFunction({T1, T2, T1}), T1, List, T2}));
    declare("fold_right", getType("fold_left"));
  }
}

llvm::SmallVector<TypeExpr*> Unifier::getParameterTypes(const std::vector<std::unique_ptr<ASTNode>>& parameters) {
  llvm::SmallVector<TypeExpr*> result;
  for (auto &param : parameters) {
    if (auto *vp = llvm::dyn_cast<ValuePatternAST>(param.get())) {
      auto id = vp->getName();
      auto *tv = createTypeVariable();
      concreteTypes.insert(tv);
      declare(id, tv);
      result.push_back(tv);
    } else if (auto *_ = llvm::dyn_cast<UnitExpressionAST>(param.get())) {
      result.push_back(getUnitType());
    } else if (auto *tp = llvm::dyn_cast<TypedPatternAST>(param.get())) {
      auto *type = getType(tp->getType()->getPath());
      auto *typeName = tp->getPattern();
      auto *vp = llvm::dyn_cast<ValuePatternAST>(typeName);
      assert(vp && "Expected value pattern");
      auto id = vp->getName();
      declare(id, type);
      result.push_back(type);
    } else {
      DBGS("unknown parameter:\n" << *param << '\n');
      assert(false && "unknown parameter");
    }
  }
  return result;
}

static llvm::SmallVector<ASTNode*> flattenProductExpression(const ProductExpressionAST* pe) {
  llvm::SmallVector<ASTNode*> result;
  for (auto &element : pe->getElements()) {
    if (auto *tp = llvm::dyn_cast<ProductExpressionAST>(element.get())) {
      auto elements = flattenProductExpression(tp);
      result.insert(result.end(), elements.begin(), elements.end());
    } else {
      result.push_back(element.get());
    }
  }
  return result;
}

static llvm::SmallVector<ValuePatternAST*> flattenTuplePattern(const TuplePatternAST* tp) {
  llvm::SmallVector<ValuePatternAST*> result;
  for (auto &element : tp->getElements()) {
    if (auto *tp = llvm::dyn_cast<TuplePatternAST>(element.get())) {
      auto elements = flattenTuplePattern(tp);
      result.insert(result.end(), elements.begin(), elements.end());
    } else if (auto *vp = llvm::dyn_cast<ValuePatternAST>(element.get())) {
      result.push_back(vp);
    } else {
      DBGS("unknown pattern: " << *element << '\n');
      assert(false && "unknown pattern");
    }
  }
  return result;
}

TypeExpr *Unifier::declarePatternVariables(const ASTNode *ast, llvm::SmallVector<TypeExpr*>& typevars) {
  if (auto *vp = llvm::dyn_cast<ValuePatternAST>(ast)) {
    typevars.push_back(createTypeVariable());
    declare(vp->getName(), typevars.back());
  } else if (auto *tp = llvm::dyn_cast<TuplePatternAST>(ast)) {
    auto elements = flattenTuplePattern(tp);
    auto tupleTypeVars = llvm::map_to_vector(elements, [&](auto *vp) -> TypeExpr* {
      auto *tv = createTypeVariable();
      declare(vp->getName(), tv);
      return tv;
    });
    typevars.push_back(createTuple(tupleTypeVars));
  } else if (auto *pp = llvm::dyn_cast<ParenthesizedPatternAST>(ast)) {
    return declarePatternVariables(pp->getPattern(), typevars);
  } else {
    DBGS("unknown pattern: " << *ast << '\n');
    assert(false && "unknown pattern");
  }
  return createTuple(typevars);
}

TypeExpr* Unifier::inferType(const ASTNode* ast) {
  DBGS('\n' << *ast << '\n');
  if (auto *n = llvm::dyn_cast<NumberExprAST>(ast)) {
    DBGS("number\n");
    llvm::StringRef value = n->getValue();
    return value.contains('.') ? getType("float") : getType("int");
  } else if (auto *_ = llvm::dyn_cast<StringExprAST>(ast)) {
    DBGS("string\n");
    return getType("string");
  } else if (auto *_ = llvm::dyn_cast<BooleanExprAST>(ast)) {
    DBGS("boolean\n");
    return getType("bool");
  } else if (auto *se = llvm::dyn_cast<SignExpressionAST>(ast)) {
    DBGS("sign\n");
    return infer(se->getOperand());
  } else if (auto *vp = llvm::dyn_cast<ValuePathAST>(ast)) {
    DBGS("value path\n");
    return getType(vp->getPath());
  } else if (auto *_ = llvm::dyn_cast<UnitExpressionAST>(ast)) {
    DBGS("unit expression\n");
    return getUnitType();
  } else if (auto *cu = llvm::dyn_cast<CompilationUnitAST>(ast)) {
    DBGS("compilation unit\n");
    EnvScope es(env);
    initializeEnvironment();
    TypeExpr *last = getUnitType();
    for (auto &item : cu->getItems()) {
      last = infer(item.get());
    }
    return last;
  } else if (auto *ae = llvm::dyn_cast<ArrayExpressionAST>(ast)) {
    DBGS("array expression\n");
    auto *elementType = createTypeVariable();
    for (auto &element : ae->getElements()) {
      unify(infer(element.get()), elementType);
    }
    return getListOf(elementType);
  } else if (auto *ei = llvm::dyn_cast<ExpressionItemAST>(ast)) {
    DBGS("expression item\n");
    return infer(ei->getExpression());
  } else if (auto *le = llvm::dyn_cast<LetExpressionAST>(ast)) {
    DBGS("let expression\n");
    EnvScope es(env);
    auto savedTypes = concreteTypes;
    infer(le->getBinding());
    auto *result = infer(le->getBody());
    concreteTypes = savedTypes;
    return result;
  } else if (auto *pe = llvm::dyn_cast<ParenthesizedExpressionAST>(ast)) {
    DBGS("parenthesized expression\n");
    return infer(pe->getExpression());
  } else if (auto *vd = llvm::dyn_cast<ValueDefinitionAST>(ast)) {
    DBGS("value definition\n");
    TypeExpr *result = getUnitType();
    for (auto &binding : vd->getBindings()) {
      result = infer(binding.get());
    }
    return result;
  } else if (auto *fe = llvm::dyn_cast<FunExpressionAST>(ast)) {
    DBGS("function expression\n");
    EnvScope es(env);
    auto savedTypes = concreteTypes;
    auto argTypes = getParameterTypes(fe->getParameters());
    auto *resultType = createTypeVariable();
    argTypes.push_back(resultType);
    auto *functionType = createFunction(argTypes);
    unify(resultType, infer(fe->getBody()));
    concreteTypes = savedTypes;
    return functionType;
  } else if (auto *lb = llvm::dyn_cast<LetBindingAST>(ast)) {
    DBGS("let binding\n");
    const auto &parameters = lb->getParameters();
    if (parameters.empty()) {
      auto *bodyType = infer(lb->getBody());
      declare(lb->getName(), bodyType);
      return bodyType;
    } else {
      TypeExpr *functionType;
      const bool isRecursive = lb->getIsRecursive();
      auto name = lb->getName();
      if (isRecursive) {
        DBGS("Recursive let binding: " << name << '\n');
        declare(name, createTypeVariable());
      }
      {
        EnvScope es(env);
        auto savedTypes = concreteTypes;
        auto types = getParameterTypes(parameters);
        auto *bodyType = infer(lb->getBody());
        types.push_back(bodyType);
        concreteTypes = savedTypes;
        functionType = createFunction(types);
      }
      DBGS("declared function type for " << name << ": " << *functionType << '\n');
      if (isRecursive) {
        unify(functionType, getType(name));
      } else {
        declare(name, functionType);
      }
      DBGS("inferred function type for " << name << ": " << *functionType << '\n');
      return functionType;
    }
  } else if (auto *le = llvm::dyn_cast<ListExpressionAST>(ast)) {
    auto *elementType = createTypeVariable();
    for (auto &element : le->getElements()) {
      unify(infer(element.get()), elementType);
    }
    return getListOf(elementType);
  } else if (auto *ag = llvm::dyn_cast<ArrayGetExpressionAST>(ast)) {
    DBGS("array get\n");
    auto *index = infer(ag->getIndex());
    unify(index, getIntType());
    auto *inferredListType = infer(ag->getArray());
    auto *listType = getListOf(createTypeVariable());
    unify(inferredListType, listType);
    return listType->at(0);
  } else if (auto *ite = llvm::dyn_cast<IfExpressionAST>(ast)) {
    DBGS("if expression\n");
    auto *cond = ite->getCondition();
    auto *condType = infer(cond);
    unify(condType, getType("bool"));

    auto *thenAst = ite->getThenBranch();
    auto *thenType = infer(thenAst);

    if (ite->hasElseBranch()) {
      auto *elseAst = ite->getElseBranch();
      auto *elseType = infer(elseAst);
      unify(thenType, elseType);
    }

    return thenType;
  } else if (auto *ie = llvm::dyn_cast<InfixExpressionAST>(ast)) {
    DBGS("infix expression\n");
    auto *left = infer(ie->getLHS());
    auto *right = infer(ie->getRHS());
    auto *operationType = getType(ie->getOperator());
    auto *functionType = createFunction({left, right, createTypeVariable()});
    unify(functionType, operationType);
    return functionType->back();
  } else if (auto *ae = llvm::dyn_cast<ApplicationExprAST>(ast)) {
    DBGS("application expression\n");
    auto *declaredFunctionType = infer(ae->getFunction());
    DBGS("declared function type: " << *declaredFunctionType << '\n');
    llvm::SmallVector<TypeExpr*> args;
    for (auto &arg : ae->getArguments()) {
      args.push_back(infer(arg.get()));
    }
    args.push_back(createTypeVariable()); // return type
    auto *functionType = createFunction(args);
    DBGS("function type: " << *functionType << '\n');
    unify(declaredFunctionType, functionType);
    DBGS("function type after unification: " << *functionType << '\n');
    return functionType->back();
  } else if (auto *cp = llvm::dyn_cast<ConstructorPathAST>(ast)) {
    auto name = hashPath(cp->getPath());
    DBGS("constructor path: " << name << '\n');
    auto *ctorType = getType(name);
    DBGS("ctor type: " << *ctorType << '\n');
    if (auto *op = llvm::dyn_cast<TypeOperator>(ctorType)) {
      if (op->getArgs().size() == 1) {
        DBGS("HACK: unwrap constructor\n");
        // TODO:
        // work around bug in treesitter upstream where constructors with no arguments
        // are not wrapped in an application expression... so we unwrap our selves.
        return op->back();
      }
    }
    return ctorType;
  } else if (auto *cp = llvm::dyn_cast<ConstructorPatternAST>(ast)) {
    DBGS("constructor pattern\n");
    auto &args = cp->getArguments();
    llvm::SmallVector<TypeExpr*> typevars;
    llvm::for_each(args, [&](auto &arg) {
      declarePatternVariables(arg.get(), typevars);
    });
    typevars.push_back(createTypeVariable()); // return type
    auto *functionType = createFunction({typevars});
    auto *ctorType = getType(cp->getConstructor()->getPath());
    unify(functionType, ctorType);
    return functionType->back();
  } else if (auto *vp = llvm::dyn_cast<ValuePatternAST>(ast)) {
    DBGS("value pattern\n");
    return getType(vp->getName());
  } else if (auto *gp = llvm::dyn_cast<GuardedPatternAST>(ast)) {
    DBGS("guarded pattern\n");
    auto *pattern = gp->getPattern();
    auto *guard = gp->getGuard();
    auto *patternType = infer(pattern);
    auto *guardType = infer(guard);
    unify(guardType, getType("bool"));
    return patternType;
  } else if (auto *me = llvm::dyn_cast<MatchExpressionAST>(ast)) {
    DBGS("match expression\n");
    auto *value = me->getValue();
    auto &cases = me->getCases();
    auto *valueType = infer(value);
    auto *resultType = createTypeVariable();
    for (auto &caseAst : cases) {
      EnvScope es(env);
      auto savedTypes = concreteTypes;
      auto *pattern = caseAst->getPattern();
      // inference on a pattern will check the guard, if there is one
      auto *patternType = infer(pattern);
      // TODO: will the pattern type always match the value type?
      unify(valueType, patternType);
      // 
      auto *expressionType = infer(caseAst->getExpression());
      unify(expressionType, resultType);
      concreteTypes = savedTypes;
    }
    return resultType;
  } else if (auto *td = llvm::dyn_cast<TypeDefinitionAST>(ast)) {
    DBGS("type definition\n");
    for (auto &binding : td->getBindings()) {
      auto typeName = binding->getName();
      declare(typeName, createTypeOperator(typeName));
      auto *definition = binding->getDefinition();
      if (auto *vd = llvm::dyn_cast<VariantDeclarationAST>(definition)) {
        for (auto &ctor : vd->getConstructors()) {
          auto ctorName = ctor->getName();
          llvm::SmallVector<TypeExpr *> args =
              llvm::map_to_vector(ctor->getOfTypes(), [&](auto &ofType) {
                return getType(ofType->getPath());
              });
          TypeExpr *ctorType;
          auto *newTypeType = getType(typeName);
          if (args.size() > 1) {
            auto *tupleType = createTuple(args);
            ctorType = createFunction({tupleType, newTypeType});
          } else {
            args.push_back(getType(typeName));
            ctorType = createFunction(args);
          }
          declare(ctorName, ctorType);
        }
      } else {
        DBGS("unknown definition: " << *definition << '\n');
        assert(false && "unknown definition");
      }
    }
    return getUnitType();
  } else if (auto *pe = llvm::dyn_cast<ProductExpressionAST>(ast)) {
    DBGS("product expression\n");
    auto exprs = flattenProductExpression(pe);
    auto typevars = llvm::map_to_vector(exprs, [&](auto *expr) -> TypeExpr* {
      return infer(expr);
    });
    return createTuple(typevars);
  } else if (auto *se = llvm::dyn_cast<SequenceExpressionAST>(ast)) {
    DBGS("sequence expression\n");
    for (auto expr : llvm::enumerate(se->getExpressions())) {
      auto *result = infer(expr.value().get());
      if (expr.index() == se->getNumExpressions() - 1) {
        return result;
      }
    }
    assert(false && "sequence expression should have at least one expression");
  } else if (auto *mi = llvm::dyn_cast<ModuleImplementationAST>(ast)) {
    DBGS("module implementation\n");
    auto &items = mi->getItems();
    return std::accumulate(
        items.begin(), items.end(), getUnitType(),
        [&](auto *_, auto &item) {
          return infer(item.get());
        });
  } else if (auto *md = llvm::dyn_cast<ModuleDefinitionAST>(ast)) {
    DBGS("module definition\n");
    ModuleScope ms(*this, md->getName());
    if (auto *sig = md->getSignature()) {
      infer(sig);
    } else if (auto *impl = md->getImplementation()) {
      infer(impl);
    } else {
      assert(false && "module definition must have either a signature or implementation");
    }
    auto path = std::vector<std::string>{"Module", md->getName()};
    auto str = stringArena.save(hashPath(path));
    DBGS("Declared module: " << str << '\n');
    return createTypeOperator(str);
  } else if (auto *fe = llvm::dyn_cast<ForExpressionAST>(ast)) {
    DBGS("for expression\n");
    EnvScope es(env);
    auto savedTypes = concreteTypes;
    auto loopVar = fe->getLoopVar();
    declare(loopVar, getType("int"));
    auto *start = infer(fe->getStartExpr());
    unify(start, getType("int"));
    auto *end = infer(fe->getEndExpr());
    unify(end, getType("int"));
    auto *body = infer(fe->getBody());
    unify(body, getUnitType());
    concreteTypes = savedTypes;
    return getUnitType();
  } else {
    DBGS("Unknown AST node: " << ASTNode::getName(*ast) << '\n');
    assert(false && "Unknown AST node");
  }
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
void Unifier::unify(TypeExpr* a, TypeExpr* b) {
  static size_t count = 0;
  auto *wildcard = getWildcardType();
  a = prune(a);
  b = prune(b);
  DBGS("Unifying:" << count++ << ": " << *a << " and " << *b << '\n');
  if (auto *tva = llvm::dyn_cast<TypeVariable>(a)) {
    if (auto *tvb = llvm::dyn_cast<TypeVariable>(b);
        tvb && isConcrete(tva, concreteTypes) &&
        isGeneric(tvb, concreteTypes)) {
      unify(tvb, tva);
      return;
    }
    if (*a != *b) {
      DBGS("Unifying type variable with different type\n");
      if (isSubType(a, b)) {
        assert(false && "Recursive unification");
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
        return;
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
          unify(toa->back(), tob->back());
          return;
        }
        if (isWildcard(aa) or isWildcard(bb)) {
          DBGS("Unifying with wildcard, skipping unification for this element type\n");
          continue;
        }
        unify(aa, bb);
      }
    }
  }
}

static size_t clone_count = 0;
TypeExpr *Unifier::clone(TypeExpr *type) {
  DBGS("Cloning type: " << ++clone_count << ": " << *type << '\n');
  llvm::DenseMap<TypeExpr *, TypeExpr *> mapping;
  auto *unifier = this;
  auto recurse = [&](this const auto &recurse, TypeExpr *expr) -> TypeExpr * {
    DBGS("recursing on type: " << *expr << '\n');
    if (auto *op = llvm::dyn_cast<TypeOperator>(expr)) {
      auto args = llvm::to_vector(llvm::map_range(op->getArgs(), [&](TypeExpr* arg) {
        return unifier->clone(arg);
      }));
      DBGS("cloning type operator: " << op->getName() << '\n');
      return unifier->createTypeOperator(op->getName(), args);
    } else if (auto *tv = llvm::dyn_cast<TypeVariable>(expr)) {
      DBGS("cloning type variable: " << *tv << '\n');
      if (unifier->isGeneric(tv, concreteTypes)) {
        DBGS("type variable is generic, cloning\n");
        if (mapping.find(tv) == mapping.end()) {
          DBGS("Didn't find mapping for type variable: " << *tv << ", creating new one\n");
          mapping[tv] = unifier->createTypeVariable();
        }
        DBGS("returning cloned type variable: " << *mapping[tv] << '\n');
        return mapping[tv];
      }
    }
    DBGS("cloned type: " << *expr << '\n');
    return expr;
  };
  return recurse(prune(type));
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

} // namespace ocamlc2

