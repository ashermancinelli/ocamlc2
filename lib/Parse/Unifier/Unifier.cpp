#include "ocamlc2/Support/CL.h"
#include "ocamlc2/Parse/TypeSystem.h"
#include "ocamlc2/Parse/TSUnifier.h"
#include "ocamlc2/Parse/TSUtil.h"
#include "ocamlc2/Parse/AST.h"
#include "ocamlc2/Support/Utils.h"
#include <cstdint>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <llvm/Support/FileSystem.h>

#define DEBUG_TYPE "unifier"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"

namespace ocamlc2 {

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
  ORFAIL(a);
  ORFAIL(b);
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
      FAIL_IF(isSubType(a, b), "Recursive unification");
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
      auto fHasVarargs = [&](TypeExpr* arg) { return llvm::isa<VarargsOperator>(arg); };
      const bool hasVarargs = llvm::any_of(toa->getArgs(), fHasVarargs) or
                              llvm::any_of(tob->getArgs(), fHasVarargs);
      DBGS("has varargs: " << hasVarargs << '\n');

      if (toa->getName() == wildcard->getName() or tob->getName() == wildcard->getName()) {
        DBGS("Unifying with wildcard\n");
        return success();
      }

      if (auto *ro = llvm::dyn_cast<RecordOperator>(toa)) {
        if (auto *ro2 = llvm::dyn_cast<RecordOperator>(tob)) {
          return unifyRecordTypes(ro, ro2);
        }
      }

      // Usual type checking - if we don't have a special case, then we need operator types to
      // match in form.
      const bool sizesMatch = toa->getArgs().size() == tob->getArgs().size() or hasVarargs;
      FAIL_IF(toa->getName() != tob->getName() or not sizesMatch,
              SSWRAP("Could not unify types: " << *toa << " and " << *tob));

      for (auto [aa, bb] : llvm::zip(toa->getArgs(), tob->getArgs())) {
        ORFAIL(aa);
        ORFAIL(bb);
        if (isVarargs(aa) or isVarargs(bb)) {
          DBGS("Unifying with varargs, unifying return types and giving up\n");
          FAIL_IF(failed(unify(toa->back(), tob->back())), "failed to unify");
          return success();
        }
        if (isWildcard(aa) or isWildcard(bb)) {
          DBGS("Unifying with wildcard, skipping unification for this element type\n");
          continue;
        }
        FAIL_IF(failed(unify(aa, bb)), "failed to unify");
      }
    }
  }
  return success();
}

LogicalResult Unifier::unifyRecordTypes(RecordOperator *a, RecordOperator *b) {
  DBGS("Unifying record types: " << *a << " and " << *b << '\n');
  const auto namesmatch = a->getName() == b->getName() or a->isAnonymous() or b->isAnonymous();
  FAIL_IF(not namesmatch, SSWRAP("Could not unify record types: " << *a << " and " << *b));
  const auto aFields = a->getArgs();
  const auto bFields = b->getArgs();
  FAIL_IF(aFields.size() != bFields.size(),
          SSWRAP("Could not unify record types: " << *a << " and " << *b));
  for (auto [aa, bb] : llvm::zip(aFields, bFields)) {
    FAIL_IF(failed(unify(aa, bb)), "failed to unify");
  }
  for (auto [aa, bb] : llvm::zip(a->getFieldNames(), b->getFieldNames())) {
    FAIL_IF((aa != bb) or aa.empty() or bb.empty(),
            SSWRAP("Could not unify record types: " << *a << " and " << *b));
  }
  return success();
}

TypeExpr *Unifier::clone(TypeExpr *type) {
  llvm::DenseMap<TypeExpr *, TypeExpr *> mapping;
  return clone(type, mapping);
}

TypeExpr *Unifier::cloneOperator(TypeOperator *op, llvm::SmallVector<TypeExpr *> &mappedArgs) {
  if (auto *func = llvm::dyn_cast<FunctionOperator>(op)) {
    DBGS("Cloning function operator: " << *func << '\n');
    return getFunctionType(mappedArgs, func->parameterDescriptors);
  } else if (auto *record = llvm::dyn_cast<RecordOperator>(op)) {
    DBGS("Cloning record operator: " << *record << '\n');
    return getRecordType(record->getName(), mappedArgs, record->getFieldNames());
  } else if (auto *module = llvm::dyn_cast<ModuleOperator>(op)) {
    DBGS("Cloning module operator: " << *module << '\n');
    return create<ModuleOperator>(*module);
  } else {
    DBGS("Cloning type operator: " << *op << '\n');
    return createTypeOperator(op->getKind(), op->getName(), mappedArgs);
  }
}

#define DBGSCLONE DBGS
TypeExpr *Unifier::clone(TypeExpr *type, llvm::DenseMap<TypeExpr *, TypeExpr *> &mapping) {
  DBGSCLONE("Cloning type: " << *type << '\n');
  type = prune(type);
  DBGSCLONE("recursing on type: " << *type << '\n');
  if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    auto args =
        llvm::to_vector(llvm::map_range(op->getArgs(), [&](TypeExpr *arg) {
          return clone(arg, mapping);
        }));
    DBGSCLONE("cloning type operator: " << op->getName() << '\n');
    return cloneOperator(op, args);
  } else if (auto *tv = llvm::dyn_cast<TypeVariable>(type)) {
    DBGSCLONE("cloning type variable: " << *tv << '\n');
    if (isGeneric(tv, concreteTypes)) {
      DBGSCLONE("type variable is generic, cloning\n");
      if (mapping.find(tv) == mapping.end()) {
        DBGSCLONE("Didn't find mapping for type variable: "
            << *tv << ", creating new one\n");
        mapping[tv] = createTypeVariable();
      }
      DBGSCLONE("returning cloned type variable: " << *mapping[tv] << '\n');
      return mapping[tv];
    }
  }
  DBGSCLONE("cloned type: " << *type << '\n');
  return type;
}
#undef DBGSCLONE

TypeExpr* Unifier::prune(TypeExpr* type) {
  if (auto *tv = llvm::dyn_cast<TypeVariable>(type); tv && tv->instantiated()) {
    tv->instance = prune(tv->instance);
    return tv->instance;
  }
  return type;
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
