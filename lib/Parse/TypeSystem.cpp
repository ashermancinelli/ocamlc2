#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
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

void Unifier::declare(llvm::StringRef name, TypeExpr* type) {
  DBGS("Declaring: " << name << " as " << *type << '\n');
  auto str = stringArena.save(name.str());
  if (env.count(str)) {
    assert(false && "Type already declared");
  }
  env.insert(str, type);
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
  DBGS("Inferred type: " << *type << '\n');
  return type;
}

void Unifier::initializeEnvironment() {
  for (auto name : {"int", "float", "bool", "string", "unit", "_"}) {
    declare(name, createTypeOperator(name));
  }
  auto *T_bool = getBoolType();
  auto *T_float = getFloatType();
  auto *T_int = getIntType();
  auto *T_unit = getUnitType();
  auto *T1 = createTypeVariable();
  for (auto arithmetic : {"+", "-", "*", "/", "%"}) {
    declare(arithmetic, createFunction({T_int, T_int, T_int}));
    declare(std::string(arithmetic) + ".", createFunction({T_float, T_float, T_float}));
  }
  for (auto comparison : {"==", "!=", "<", "<=", ">", ">="}) {
    declare(comparison, createFunction({T1, T1, T_bool}));
  }
  declare("print_int", createFunction({T_int, T_unit}));
  declare("length", createFunction({createTypeVariable(), T_int}));

  // Builtin constructors
  auto *T = createTypeVariable();
  auto *Optional = createTypeOperator("Optional", {T});
  declare("None", createFunction({T_unit, Optional}));
  declare("Some", createFunction({T, Optional}));

  auto *List = createTypeOperator("List", {T});
  declare("Nil", createFunction({T_unit, List}));
  declare("Cons", createFunction({T, List, List}));
}

#if 0
MatchExpr:
| Value:
| | ValuePath: n
| Cases:
| | MatchCase:
| | | Pattern:
| | | | Number: 0
| | | Expression:
| | | | String: "Zero"
| | MatchCase:
| | | Pattern:
| | | | Number: 1
| | | Expression:
| | | | String: "One"
| | MatchCase:
| | | Pattern:
| | | | Number: 2
| | | Expression:
| | | | String: "Two"
| | MatchCase:
| | | Pattern:
| | | | GuardedPattern:
| | | | | Pattern:
| | | | | | ValuePattern: n
| | | | | Guard:
| | | | | | InfixExpr: >
| | | | | | | LHS:
| | | | | | | | ValuePath: n
| | | | | | | RHS:
| | | | | | | | Number: 0
| | | Expression:
| | | | String: "Positive"
#endif
static std::string getPath(llvm::ArrayRef<std::string> path) {
  return llvm::join(path, ".");
}

TypeExpr* Unifier::inferType(const ASTNode* ast) {
  DBGS('\n' << *ast << '\n');
  if (auto *_ = llvm::dyn_cast<NumberExprAST>(ast)) {
    return getType("int");
  } else if (auto *_ = llvm::dyn_cast<StringExprAST>(ast)) {
    return getType("string");
  } else if (auto *vp = llvm::dyn_cast<ValuePathAST>(ast)) {
    return getType(getPath(vp->getPath()));
  } else if (auto *cu = llvm::dyn_cast<CompilationUnitAST>(ast)) {
    EnvScope es(env);
    initializeEnvironment();
    TypeExpr *last = nullptr;
    for (auto &item : cu->getItems()) {
      last = infer(item.get());
    }
    return last ? last : getType("unit");
  } else if (auto *ei = llvm::dyn_cast<ExpressionItemAST>(ast)) {
    return infer(ei->getExpression());
  } else if (auto *le = llvm::dyn_cast<LetExpressionAST>(ast)) {
    EnvScope es(env);
    auto savedTypes = concreteTypes;
    infer(le->getBinding());
    auto *result = infer(le->getBody());
    concreteTypes = savedTypes;
    return result;
  } else if (auto *pe = llvm::dyn_cast<ParenthesizedExpressionAST>(ast)) {
    return infer(pe->getExpression());
  } else if (auto *vd = llvm::dyn_cast<ValueDefinitionAST>(ast)) {
    for (auto &binding : vd->getBindings()) {
      infer(binding.get());
    }
    return getType("unit");
  } else if (auto *lb = llvm::dyn_cast<LetBindingAST>(ast)) {
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
  } else if (auto *ite = llvm::dyn_cast<IfExpressionAST>(ast)) {
    auto *cond = ite->getCondition();
    auto *thenAst = ite->getThenBranch();
    auto *elseAst = ite->hasElseBranch() ? ite->getElseBranch() : nullptr;
    auto *condType = infer(cond);
    unify(condType, getType("bool"));
    auto *thenType = infer(thenAst);
    auto *elseType = infer(elseAst);
    unify(thenType, elseType);
    return thenType;
  } else if (auto *ie = llvm::dyn_cast<InfixExpressionAST>(ast)) {
    auto *left = infer(ie->getLHS());
    auto *right = infer(ie->getRHS());
    auto *operationType = getType(ie->getOperator());
    auto *functionType = createFunction({left, right, createTypeVariable()});
    unify(functionType, operationType);
    return functionType->back();
  } else if (auto *ae = llvm::dyn_cast<ApplicationExprAST>(ast)) {
    auto *declaredFunctionType = infer(ae->getFunction());
    auto args = llvm::map_to_vector(ae->getArguments(), [&](auto &arg) {
      return infer(arg.get());
    });
    args.push_back(createTypeVariable()); // return type
    auto *functionType = createFunction(args);
    unify(functionType, declaredFunctionType);
    return functionType->back();
  // } else if (auto *cp = llvm::dyn_cast<ConstructorPathAST>(ast)) {
  //   auto name = getPath(cp->getPath());
  //   auto *constructorType = getType(name);
  //   if (constructorType->getArgs().size() != cp->getArguments().size()) {
  //     llvm::errs() << "Constructor " << name << " expects " << constructorType->getArgs().size() << " arguments, but got " << cp->getArguments().size() << '\n';
  //     assert(false && "Constructor expects wrong number of arguments");
  //   }
  //   auto args = llvm::map_to_vector(ae->getArguments(), [&](auto &arg) {
  //     return infer(arg.get());
  //   });
  } else if (auto *vp = llvm::dyn_cast<ValuePatternAST>(ast)) {
    return getType(vp->getName());
  } else if (auto *gp = llvm::dyn_cast<GuardedPatternAST>(ast)) {
    auto *pattern = gp->getPattern();
    auto *guard = gp->getGuard();
    auto *patternType = infer(pattern);
    auto *guardType = infer(guard);
    unify(guardType, getType("bool"));
    return patternType;
  } else if (auto *me = llvm::dyn_cast<MatchExpressionAST>(ast)) {
    auto *value = me->getValue();
    auto &cases = me->getCases();
    auto *valueType = infer(value);
    auto *resultType = createTypeVariable();
    for (auto &caseAst : cases) {
      auto *pattern = caseAst->getPattern();
      // inference on a pattern will check the guard, if there is one
      auto *patternType = infer(pattern);
      // TODO: will the pattern type always match the value type?
      unify(valueType, patternType);
      // 
      auto *expressionType = infer(caseAst->getExpression());
      unify(expressionType, resultType);
    }
    return resultType;
  } else if (auto *fe = llvm::dyn_cast<ForExpressionAST>(ast)) {
    EnvScope es(env);
    auto savedTypes = concreteTypes;
    auto loopVar = fe->getLoopVar();
    declare(loopVar, getType("int"));
    auto *start = infer(fe->getStartExpr());
    unify(start, getType("int"));
    auto *end = infer(fe->getEndExpr());
    unify(end, getType("int"));
    auto *body = infer(fe->getBody());
    unify(body, getType("unit"));
    concreteTypes = savedTypes;
    return getType("unit");
  } else {
    DBGS("Unknown AST node: " << ASTNode::getName(*ast) << '\n');
    assert(false && "Unknown AST node");
  }
  return nullptr;
}

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
      if (toa->getName() == wildcard->getName() or tob->getName() == wildcard->getName()) {
        DBGS("Unifying with wildcard\n");
        return;
      }
      if (toa->getName() != tob->getName() or toa->getArgs().size() != tob->getArgs().size()) {
        llvm::errs() << "Could not unify types: " << *toa << " and " << *tob << '\n';
        assert(false);
      }
      for (auto [aa, bb] : llvm::zip(toa->getArgs(), tob->getArgs())) {
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

} // namespace ocamlc2

