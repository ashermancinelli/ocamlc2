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
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/raw_ostream.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <llvm/Support/FileSystem.h>

#define DEBUG_TYPE "Unifier.cpp"
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
  auto result = doUnify(a, b);
  return result;
}

LogicalResult Unifier::doUnify(TypeExpr* a, TypeExpr* b) {
  ORFAIL(a);
  ORFAIL(b);
  static size_t count = 0;
  auto *wildcard = getWildcardType();
  a = prune(a);
  b = prune(b);
  // if (auto *alias = llvm::dyn_cast<TypeAlias>(a)) {
  //   return unify(alias->getType(), b);
  // }
  // if (auto *alias = llvm::dyn_cast<TypeAlias>(b)) {
  //   return unify(a, alias->getType());
  // }
  DBGS("Unifying:" << count++ << ": " << *a << " and " << *b << '\n');
  if (auto *tva = llvm::dyn_cast<TypeVariable>(a)) {
    if (auto *tvb = llvm::dyn_cast<TypeVariable>(b);
        tvb && isConcrete(tva, concreteTypes) &&
        isGeneric(tvb, concreteTypes)) {
      return unify(tvb, tva);
    }
    if (*a != *b) {
      DBGS("Unifying type variable with different type: " << *tva << "'s instance is now: " << *b << "\n");
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
      if (auto *ro = llvm::dyn_cast<RecordOperator>(toa)) {
        if (auto *ro2 = llvm::dyn_cast<RecordOperator>(tob)) {
          return unifyRecordTypes(ro, ro2);
        }
      }

      auto *soa = llvm::dyn_cast<SignatureOperator>(toa);
      auto *sob = llvm::dyn_cast<SignatureOperator>(tob);
      if (soa or sob) {
        FAIL_IF(not soa or not sob, SSWRAP("Can not unify signature with non-signature"));
        return unifySignatureTypes(soa, sob);
      }

      auto *foa = llvm::dyn_cast<FunctorOperator>(toa);
      auto *fob = llvm::dyn_cast<FunctorOperator>(tob);
      if (foa or fob) {
        FAIL_IF(not foa or not fob, SSWRAP("Can not unify functor with non-functor"));
        return unifyFunctorTypes(foa, fob);
      }

      // Check if we have any varargs types in the arguments - if so, the forms are allowed
      // to be different.
      auto fHasVarargs = [&](TypeExpr* arg) { return llvm::isa<VarargsOperator>(arg); };
      const bool hasVarargs = llvm::any_of(toa->getArgs(), fHasVarargs) or
                              llvm::any_of(tob->getArgs(), fHasVarargs);
      // DBGS("has varargs: " << hasVarargs << '\n');

      if (toa->getName() == wildcard->getName() or tob->getName() == wildcard->getName()) {
        DBGS("Unifying with wildcard\n");
        return success();
      }

      // Usual type checking - if we don't have a special case, then we need operator types to
      // match in form.
      const bool sizesMatch = toa->getArgs().size() == tob->getArgs().size() or hasVarargs;
      const bool namesMatch = toa->getName() == tob->getName();
      const bool match = namesMatch and sizesMatch;
      FAIL_IF(not match, SSWRAP("Could not unify types: " << *toa << " and " << *tob));

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

LogicalResult Unifier::unifyNames(SignatureOperator *a, SignatureOperator *b) {
  DBGS("Unifying signature with signature:\n" << *a << "\nand\n" << *b << '\n');
  FAIL_IF(not a->getArgs().empty() and not b->getArgs().empty(),
          SSWRAP("One signature must be empty to unify two plain signatures"));
  
  enum ExportKind { Local, Exported };
  const SmallVector<std::tuple<ArrayRef<SignatureOperator::Export>, ExportKind, SignatureOperator *>> plan{
    {a->getExports(), ExportKind::Exported, b},
    {a->getLocals(), ExportKind::Local, b},
    {b->getExports(), ExportKind::Exported, a},
    {b->getLocals(), ExportKind::Local, a},
  };
  
  for (auto [exports, localOrExport, signature] : plan) {
    if (signature->isModuleType())
      continue;
    for (auto exported : exports) {
      auto name = exported.name;
      auto kind = exported.kind;
      auto type = exported.type;
      DBGS("Unifying exported name: " << name << " kind: " << kind
                                      << " type: " << *type << '\n');
      auto *tv = createTypeVariable();
      switch (kind) {
      case SignatureOperator::Export::Type: {
        if (!signature->lookupType(name)) {
          if (localOrExport == ExportKind::Exported) {
            signature->exportType(name, tv);
          } else {
            signature->localType(name, tv);
          }
        }
        FAIL_IF(failed(unify(a->lookupType(name), b->lookupType(name))),
                "failed to unify");
        break;
      }
      case SignatureOperator::Export::Variable: {
        if (!signature->lookupVariable(name)) {
          if (localOrExport == ExportKind::Exported) {
            signature->exportVariable(name, tv);
          } else {
            signature->localVariable(name, tv);
          }
        }
        FAIL_IF(failed(unify(a->lookupVariable(name), b->lookupVariable(name))),
                "failed to unify");
        break;
      }
      case SignatureOperator::Export::Exception: {
        assert(false && "NYI");
        break;
      }
      }
    }
  }

  return success();
}

LogicalResult Unifier::unifyModuleWithSignature(ModuleOperator *module, SignatureOperator *signature) {
  DBGS("Unifying module with signature:\n" << *module << "\nand\n" << *signature << '\n');
  FAIL_IF(module->getExports().size() < signature->getExports().size(),
          SSWRAP("Could not unify module with signature: " << *module << " and " << *signature));
  for (auto exported : signature->getExports()) {
    auto name = exported.name;
    auto kind = exported.kind;
    auto type = exported.type;
    DBGS("Unifying exported name: " << name << " kind: " << kind << " type: " << *type << '\n');
    switch (kind) {
    case SignatureOperator::Export::Type: {
      DBGS("Unifying type decl\n");
      auto *matched = module->lookupType(name);
      FAIL_IF(not matched, SSWRAP("Could not find type "
                                  << name << " in module " << *module));
      FAIL_IF(failed(unify(matched, type)), "failed to unify");
      break;
    }
    case SignatureOperator::Export::Variable: {
      DBGS("Unifying variable type\n");
      FAIL_IF(not module->lookupVariable(name),
              SSWRAP("Could not find variable " << name << " in module "
                                                << *module));
      FAIL_IF(failed(unify(module->lookupVariable(name), type)),
              "failed to unify");
      break;
    }
    case SignatureOperator::Export::Exception:
      assert(false && "NYI");
      break;
    }
  }

  return success();
}

static bool namesMatch(FunctorOperator *a, FunctorOperator *b) {
  const auto nameA = a->getName(), nameB = b->getName();
  const bool anyAnon = nameA.empty() or nameB.empty();
  const bool namesMatch = anyAnon or nameA == nameB;
  DBGS("namesMatch: " << namesMatch << " (" << nameA << " =~ " << nameB << ")\n");
  return namesMatch;
}

LogicalResult Unifier::unifyFunctorTypes(FunctorOperator *a, FunctorOperator *b) {
  DBGS("Unifying functor types:\n" << *a << "\nand\n" << *b << '\n');
  FAIL_IF(a->getArgs().size() != b->getArgs().size(),
          SSWRAP("Could not unify functors: " << *a << " and " << *b));
  FAIL_IF(not namesMatch(a, b), SSWRAP("Could not unify functors with different names: " << *a << " and " << *b));
  for (auto [aa, bb] : llvm::zip(a->getArgs(), b->getArgs())) {
    FAIL_IF(failed(unify(aa, bb)), "failed to unify");
  }
  return success();
}

LogicalResult Unifier::unifySignatureTypes(SignatureOperator *a, SignatureOperator *b) {
  DBGS("Unifying signature types:\n" << *a << "\nand\n" << *b << '\n');
  return unifyNames(a, b);
}

LogicalResult Unifier::unifyRecordTypes(RecordOperator *a, RecordOperator *b) {
  DBGS("Unifying record types: " << *a << " and " << *b << '\n');
  const auto namesmatch = a->getName() == b->getName() or a->isAnonymous() or b->isAnonymous();
  FAIL_IF(not namesmatch, SSWRAP("Could not unify record types: " << *a << " and " << *b));
  const auto aTypeArgs = a->getArgs();
  const auto bTypeArgs = b->getArgs();
  FAIL_IF(aTypeArgs.size() != bTypeArgs.size(),
          SSWRAP("Could not unify record types: " << *a << " and " << *b));
  for (auto [aa, bb] : llvm::zip(aTypeArgs, bTypeArgs)) {
    FAIL_IF(failed(unify(aa, bb)), "failed to unify");
  }
  const auto aFields = a->getFieldTypes();
  const auto bFields = b->getFieldTypes();
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
  auto *cloned = clone(type, mapping);
  DBGS("Cloned type: " << *cloned << '\n');
  return cloned;
}

TypeOperator *Unifier::cloneOperatorWithoutMutuallyRecursiveTypes(TypeOperator *op, llvm::SmallVector<TypeExpr *> &mappedArgs, llvm::DenseMap<TypeExpr *, TypeExpr *> &mapping) {
  if (auto *func = llvm::dyn_cast<FunctionOperator>(op)) {
    DBGS("Cloning function operator: " << *func << '\n');
    for (auto [arg, mappedArg] : llvm::zip(func->getArgs(), mappedArgs  )) {
      DBGS("Cloning argument: " << *arg << " to " << *mappedArg << '\n');
    }
    auto *newFunc = getFunctionType(mappedArgs, func->parameterDescriptors);
    DBGS("Cloned function operator: " << *newFunc << '\n');
    return newFunc;
  } else if (auto *record = llvm::dyn_cast<RecordOperator>(op)) {
    DBGS("Cloning record operator: " << *record << '\n');
    auto fieldNames = record->getFieldNames();
    auto fieldTypes = record->getFieldTypes();
    auto newFieldTypes = llvm::to_vector(llvm::map_range(fieldTypes, [&](TypeExpr *arg) {
      return clone(arg, mapping);
    }));
    auto *r = create<RecordOperator>(record->getName(), mappedArgs, newFieldTypes, fieldNames);
    DBGS("Cloned record operator: " << *r << '\n');
    return r;
  } else if (auto *variant = llvm::dyn_cast<VariantOperator>(op)) {
    DBGS("Cloning variant operator: " << *variant << '\n');
    auto *newVariant = create<VariantOperator>(variant->getName(), mappedArgs);
    for (auto ctor : variant->getConstructors()) {
      if (std::holds_alternative<
              std::pair<llvm::StringRef, FunctionOperator *>>(ctor)) {
        auto [name, func] =
            std::get<std::pair<llvm::StringRef, FunctionOperator *>>(ctor);
        auto *newArg = clone(func->getArgs().front(), mapping);
        auto *newFunc =
            getFunctionType({newArg, newVariant}, func->parameterDescriptors);
        newVariant->addConstructor(name, newFunc);
      } else {
        newVariant->addConstructor(std::get<llvm::StringRef>(ctor));
      }
    }
    return newVariant;
  } else if (auto *functor = llvm::dyn_cast<FunctorOperator>(op)) {
    DBGS("Cloning functor operator: " << *functor << '\n');
    auto mappedParams = llvm::map_to_vector(functor->getModuleParameters(), [&](const std::pair<llvm::StringRef, SignatureOperator *> &param) {
      auto *clonedParam = clone(param.second, mapping);
      auto *casted = llvm::cast<SignatureOperator>(clonedParam);
      return std::make_pair(param.first, casted);
    });
    return create<FunctorOperator>(functor->getName(), mappedArgs, mappedParams);
  } else if (auto *signature = llvm::dyn_cast<SignatureOperator>(op)) {
    DBGS("Cloning signature operator: " << *signature << '\n');
    auto newLocals = llvm::map_to_vector(signature->getLocals(), [&](const SignatureOperator::Export &e) {
      return SignatureOperator::Export(e.kind, e.name, clone(e.type, mapping));
    });
    auto newExports = llvm::map_to_vector(signature->getExports(), [&](const SignatureOperator::Export &e) {
      return SignatureOperator::Export(e.kind, e.name, clone(e.type, mapping));
    });
    if (auto *module = llvm::dyn_cast<ModuleOperator>(op)) {
      DBGS("Cloning module operator: " << *module << '\n');
      auto *clonedModule = create<ModuleOperator>(module->getName(), mappedArgs, newExports, newLocals);
      // TODO: propagate open modules...
      return clonedModule;
    }
    return create<SignatureOperator>(signature->getName(), mappedArgs, newExports, newLocals);
  } else {
    DBGS("Cloning type operator: " << *op << '\n');
    return createTypeOperator(op->getKind(), op->getName(), mappedArgs);
  }
}

TypeOperator *Unifier::cloneOperator(TypeOperator *op, llvm::SmallVector<TypeExpr *> &mappedArgs, llvm::DenseMap<TypeExpr *, TypeExpr *> &mapping) {
  DBGS("Cloning operator: " << *op << '\n');
  auto *cloned = cloneOperatorWithoutMutuallyRecursiveTypes(op, mappedArgs, mapping);
  // cloned->addMutuallyRecursiveTypes(op->getMutuallyRecursiveTypeGroup());
  for (auto name : op->getMutuallyRecursiveTypeGroup()) {
    DBGS("Adding mutually recursive type: " << name << '\n');
    cloned->addMutuallyRecursiveType(name);
  }
  return cloned;
}

#define DBGSCLONE DBGS
TypeExpr *Unifier::clone(TypeExpr *type, llvm::DenseMap<TypeExpr *, TypeExpr *> &mapping) {
  DBGSCLONE("Cloning type: " << *type << '\n');
  type = prune(type);
  
  DBGSCLONE("recursing on type: " << *type << '\n');
  if (auto *alias = llvm::dyn_cast<TypeAlias>(type)) {
    DBGSCLONE("cloning type alias: " << alias->getName() << '\n');
    auto *cloned = create<TypeAlias>(alias->getName(), clone(alias->getType(), mapping));
    DBGSCLONE("cloned type alias: " << *cloned << '\n');
    return cloned;
  } else if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    auto args =
        llvm::to_vector(llvm::map_range(op->getArgs(), [&](TypeExpr *arg) {
          return clone(arg, mapping);
        }));
    DBGSCLONE("cloning type operator: " << op->getName() << '\n');
    return cloneOperator(op, args, mapping);
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
    DBGSCLONE("type variable is concrete, returning original\n");
  }
  DBGSCLONE("cloned type: " << *type << '\n');
  return type;
}
#undef DBGSCLONE

TypeExpr* Unifier::pruneTypeVariables(TypeExpr* type) {
  if (auto *tv = llvm::dyn_cast<TypeVariable>(type); tv && tv->instantiated()) {
    tv->instance = pruneTypeVariables(tv->instance);
    return tv->instance;
  }
  return type;
}

TypeExpr* Unifier::prune(TypeExpr* type) {
  if (auto *tv = llvm::dyn_cast<TypeVariable>(type); tv && tv->instantiated()) {
    tv->instance = prune(tv->instance);
    return tv->instance;
  }
  if (auto *alias = llvm::dyn_cast<TypeAlias>(type)) {
    return prune(alias->getType());
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
