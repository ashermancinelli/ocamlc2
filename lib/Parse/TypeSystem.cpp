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
#include <sstream>

#define DEBUG_TYPE "typesystem"
#include "ocamlc2/Support/Debug.h.inc"

namespace ocamlc2 {

struct StringArena {
  std::set<std::string> pool;
  llvm::StringRef save(std::string str) {
    auto [it, _] = pool.insert(str);
    return *it;
  }
};
static StringArena stringArena;

TypeExpr* Unifier::declare(llvm::StringRef name, TypeExpr* type) {
  DBGS("Declaring: " << name << " as " << *type << '\n');
  if (name == getWildcardType()->getName() or name == getUnitType()->getName()) {
    DBGS("probably declaring a wildcard variable in a constructor pattern or "
         "assigning to unit, skipping\n");
    return type;
  }
  auto str = stringArena.save(name.str());
  if (env.count(str)) {
    DBGS("Type of " << name << " redeclared\n");
  }
  env.insert(str, type);
  return type;
}

TypeExpr* Unifier::getType(const llvm::StringRef name) {
  if (auto *type = env.lookup(name)) {
    return clone(type);
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
    os << *var.instance;
  } else {
    os << var.getName();
  }
  return os;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeOperator& op) {
  switch (op.getArgs().size()) {
    case 0:
      os << op.getName();
      break;
    case 2:
      // Infix notation if we only have two types
      os << '(' << *op.getArgs()[0] << ' ' << op.getName() << ' ' << *op.getArgs()[1] << ')';
      break;
    default:
      os << '(' << op.getName() << ' ';
      for (auto [i, arg] : llvm::enumerate(op.getArgs())) {
        os << *arg << (i == op.getArgs().size() - 1 ? "" : " ");
      }
      os << ')';
      break;
  }
  return os;
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
  // We need to insert these directly because other type initializations require
  // varargs, wildcard, etc to define themselves.
  for (auto name : {"int", "float", "bool", "string", "unit!", "_", "•", "varargs!"}) {
    DBGS("Declaring: " << name << '\n');
    env.insert(name, createTypeOperator(name));
  }
  auto *T_bool = getBoolType();
  auto *T_float = getFloatType();
  auto *T_int = getIntType();
  auto *T_unit = getUnitType();
  auto *T_string = getStringType();
  auto *T1 = createTypeVariable();
  declare("Float.pi", T_float);
  for (auto arithmetic : {"+", "-", "*", "/", "%"}) {
    declare(arithmetic, createFunction({T_int, T_int, T_int}));
    declare(std::string(arithmetic) + ".", createFunction({T_float, T_float, T_float}));
  }
  for (auto comparison : {"=", "!=", "<", "<=", ">", ">="}) {
    declare(comparison, createFunction({T1, T1, T_bool}));
  }
  declare("sqrt", createFunction({T_float, T_float}));
  declare("print_int", createFunction({T_int, T_unit}));
  declare("print_string", createFunction({T_string, T_unit}));
  declare("Printf.printf", createFunction({T_string, getType("varargs!"), T_unit}));

  // Builtin constructors
  auto *Optional = createTypeOperator("Optional", {(T1 = createTypeVariable())});
  declare("None", createFunction({Optional}));
  declare("Some", createFunction({T1, Optional}));

  auto *List = createTypeOperator("List", {(T1 = createTypeVariable())});
  declare("Nil", createFunction({List}));
  declare("Cons", createFunction({T1, List, List}));
  declare("Array.length", createFunction({List, T_int}));
  declare("List.map", createFunction({createFunction({T1, T1}), List, List}));
  declare("List.fold_left",
          createFunction({createFunction({T1, T1, T1}), T1, List, T1}));
  declare("List.fold_right", getType("List.fold_left"));
}

static std::string getPath(llvm::ArrayRef<std::string> path) {
  return llvm::join(path, ".");
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
    return getType(getPath(vp->getPath()));
  } else if (auto *cu = llvm::dyn_cast<CompilationUnitAST>(ast)) {
    DBGS("compilation unit\n");
    EnvScope es(env);
    initializeEnvironment();
    TypeExpr *last = nullptr;
    for (auto &item : cu->getItems()) {
      last = infer(item.get());
    }
    return last ? last : getUnitType();
  } else if (auto *ae = llvm::dyn_cast<ArrayExpressionAST>(ast)) {
    DBGS("array expression\n");
    auto *elementType = createTypeVariable();
    for (auto &element : ae->getElements()) {
      unify(infer(element.get()), elementType);
    }
    return getListOfType(elementType);
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
    llvm::SmallVector<TypeExpr *> argTypes = llvm::map_to_vector(
        fe->getParameters(), [&](auto &param) -> TypeExpr * {
          auto *vp = llvm::dyn_cast<ValuePatternAST>(param.get());
          assert(vp && "Expected value pattern");
          auto *paramType = createTypeVariable();
          concreteTypes.insert(paramType);
          declare(vp->getName(), paramType);
          return paramType;
    });
    auto *resultType = createTypeVariable();
    argTypes.push_back(resultType);
    auto *functionType = createFunction(argTypes);
    unify(resultType, infer(fe->getBody()));
    concreteTypes = savedTypes;
    return functionType;
  } else if (auto *lb = llvm::dyn_cast<LetBindingAST>(ast)) {
    DBGS("let binding\n");
    if (lb->getParameters().empty()) {
      auto *bodyType = infer(lb->getBody());
      declare(lb->getName(), bodyType);
      return bodyType;
    } else {
      TypeExpr *functionType;
      const bool isRecursive = lb->getIsRecursive();
      if (isRecursive) {
        DBGS("Recursive let binding: " << lb->getName() << '\n');
        declare(lb->getName(), createTypeVariable());
      }
      {
        EnvScope es(env);
        auto savedTypes = concreteTypes;
        llvm::SmallVector<TypeExpr *> types = llvm::map_to_vector(
            lb->getParameters(), [&](auto &param) -> TypeExpr * {
              auto vp = llvm::dyn_cast<ValuePatternAST>(param.get());
              assert(vp && "Expected value pattern");
              auto id = vp->getName();
              auto *tv = createTypeVariable();
              concreteTypes.insert(tv);
              declare(id, tv);
              return tv;
            });
        auto *bodyType = infer(lb->getBody());
        types.push_back(bodyType);
        concreteTypes = savedTypes;
        functionType = createFunction(types);
      }
      if (isRecursive) {
        unify(functionType, getType(lb->getName()));
      } else {
        declare(lb->getName(), functionType);
      }
      return functionType;
    }
  } else if (auto *le = llvm::dyn_cast<ListExpressionAST>(ast)) {
    auto *elementType = createTypeVariable();
    for (auto &element : le->getElements()) {
      unify(infer(element.get()), elementType);
    }
    return getListOfType(elementType);
  } else if (auto *ag = llvm::dyn_cast<ArrayGetExpressionAST>(ast)) {
    DBGS("array get\n");
    auto *index = infer(ag->getIndex());
    unify(index, getIntType());
    auto *inferredListType = infer(ag->getArray());
    auto *listType = getListOfType(createTypeVariable());
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
    auto args = llvm::map_to_vector(ae->getArguments(), [&](auto &arg) {
      return infer(arg.get());
    });
    args.push_back(createTypeVariable()); // return type
    auto *functionType = createFunction(args);
    DBGS("function type: " << *functionType << '\n');
    DBGS("declared function type: " << *declaredFunctionType << '\n');
    unify(functionType, declaredFunctionType);
    return functionType->back();
  } else if (auto *cp = llvm::dyn_cast<ConstructorPathAST>(ast)) {
    auto name = getPath(cp->getPath());
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
    auto *ctorType = getType(getPath(cp->getConstructor()->getPath()));
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
                return getType(getPath(ofType->getPath()));
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
  } else if (auto *se = llvm::dyn_cast<SequenceExpressionAST>(ast)) {
    DBGS("sequence expression\n");
    for (auto expr : llvm::enumerate(se->getExpressions())) {
      auto *result = infer(expr.value().get());
      if (expr.index() == se->getNumExpressions() - 1) {
        return result;
      }
    }
    assert(false && "sequence expression should have at least one expression");
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
  a = prune(a);
  b = prune(b);
  DBGS("Unifying: " << *a << " and " << *b << '\n');
  if (auto *tva = llvm::dyn_cast<TypeVariable>(a)) {
    if (*a != *b) {
      if (isSubType(a, b)) {
        assert(false && "Recursive unification");
      }
      tva->instance = b;
    } else {
      // fallthrough
    }
  } else if (auto *toa = llvm::dyn_cast<TypeOperator>(a)) {
    if (llvm::isa<TypeVariable>(b)) {
      return unify(b, a);
    } else if (auto *tob = llvm::dyn_cast<TypeOperator>(b)) {

      auto *wildcard = getWildcardType();

      // Check if we have any varargs types in the arguments - if so, the forms are allowed
      // to be different.
      auto fHasVarargs = [&](TypeExpr* arg) { return isVarargs(arg); };
      const bool hasVarargs = llvm::any_of(toa->getArgs(), fHasVarargs) or
                              llvm::any_of(tob->getArgs(), fHasVarargs);

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

TypeExpr *Unifier::clone(TypeExpr *type) {
  llvm::DenseMap<TypeExpr *, TypeExpr *> mapping;
  auto *unifier = this;
  auto recurse = [&](this const auto &recurse, TypeExpr *expr) -> TypeExpr * {
    if (auto *op = llvm::dyn_cast<TypeOperator>(expr)) {
      auto args = llvm::to_vector(llvm::map_range(op->getArgs(), [&](TypeExpr* arg) {
        return recurse(arg);
      }));
      return unifier->create<TypeOperator>(op->getName(), args);
    } else if (auto *tv = llvm::dyn_cast<TypeVariable>(expr)) {
      if (unifier->isGeneric(tv, concreteTypes)) {
        if (mapping.find(tv) == mapping.end()) {
          mapping[tv] = unifier->create<TypeVariable>();
        }
        return mapping[tv];
      }
    }
    return expr;
  };
  return recurse(type);
}

TypeExpr* Unifier::prune(TypeExpr* type) {
  if (auto *tv = llvm::dyn_cast<TypeVariable>(type)) {
    if (tv->instantiated()) {
      tv->instance = prune(tv->instance);
      return tv->instance;
    }
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

} // namespace ocamlc2

