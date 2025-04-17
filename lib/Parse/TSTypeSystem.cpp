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

llvm::raw_ostream& Unifier::show(ts::Cursor cursor) {
  return dump(llvm::errs(), cursor.copy(), source);
}

TypeExpr* Unifier::inferType(Cursor ast) {
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

} // namespace ts
} // namespace ocamlc2
