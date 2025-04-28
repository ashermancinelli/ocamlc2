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

#define DEBUG_TYPE "typeutilities"
#include "ocamlc2/Support/Debug.h.inc"

#include "UnifierDebug.h"

namespace ocamlc2 {

bool Unifier::isVarargs(TypeExpr* type) {
  return llvm::isa<VarargsOperator>(type);
}

bool Unifier::isWildcard(TypeExpr* type) {
  if (auto *op = llvm::dyn_cast<TypeOperator>(type)) {
    return op->getName() == getWildcardType()->getName();
  }
  return false;
}

TypeExpr *Unifier::getBoolType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getBoolOperatorName());
  return type;
}

TypeExpr *Unifier::getFloatType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getFloatOperatorName());
  return type;
}

TypeExpr *Unifier::getIntType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getIntOperatorName());
  return type;
}

TypeExpr *Unifier::getUnitType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getUnitOperatorName());
  return type;
}

TypeExpr *Unifier::getStringType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getStringOperatorName());
  return type;
}

TypeExpr *Unifier::getWildcardType() { 
  static TypeExpr *type = getDeclaredType(TypeOperator::getWildcardOperatorName());
  return type;
}

TypeExpr *Unifier::getVarargsType() { 
  static TypeExpr *type = create<VarargsOperator>();
  return type;
}


} // namespace ocamlc2
